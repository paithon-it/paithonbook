# Il problema più semplice: i bandit a più braccia

Nel 1933, sulle pagine della rivista *Biometrika*, William R. Thompson pone una
domanda che nasce da un disagio pratico {cite}`thompson1933likelihood`. In una
sperimentazione clinica si assegnano i pazienti a due trattamenti e si aspetta
la fine per sapere quale funzioni meglio. Ma a metà strada un'idea di quale sia
il migliore già ce l'abbiamo. Metà dei pazienti continua comunque a ricevere il
trattamento che sta perdendo, e quella metà è il prezzo che si paga per essere
sicuri alla fine. Thompson si chiede se il prezzo si possa ridurre spostando
via via l'assegnazione verso il trattamento che sta andando meglio, senza per
questo smettere di raccogliere prove sull'altro.

Una risposta la diede lui stesso, e conviene anticiparla perché la storia ha
un finale. Ogni paziente va assegnato tirando a sorte, ma con un dado
truccato: il trattamento che ha più probabilità di essere il migliore esce più
spesso. Quella probabilità non la si conosce, la si stima dalle prove raccolte
fino a quel momento, e a ogni paziente in più il dado si ritrucca. Chi va
meglio riceve più pazienti, ma nessuno viene scartato finché resta un dubbio.
L'idea rimase quasi ignorata per decenni e porta il nome del suo autore,
*Thompson sampling*; è una delle ricette usate oggi per mandare più visitatori
alla versione di un sito che sta rendendo di più. Prima di arrivarci conviene
però vedere tre strade che di ogni leva tengono un numero solo: sono più
semplici da raccontare e più facili da mettere in codice.

È il **dilemma fra esplorare e sfruttare** annunciato nella panoramica del
capitolo, e qui si presenta nella forma più pura che esista, perché manca tutto
il resto: nessuno stato che cambia, nessuna conseguenza differita, nessun
merito da distribuire su una catena di mosse. Solo la tensione, nuda.

Il nome viene dallo slang americano: la macchinetta da casinò con la leva si
chiama *one-armed bandit*, il bandito con un braccio solo, perché ti deruba con
educazione. Una fila di macchinette, ognuna con una probabilità di vincita
diversa e ignota, è un **bandito a più braccia**, e in inglese *multi-armed
bandit*: nome che si incontra ovunque, e che il titolo di questa sezione
abbrevia come fanno tutti, *bandit*. La domanda è quale leva tirare, e quante
volte, sapendo che ogni tiro speso a informarsi è un tiro non speso a
guadagnare.

## Un solo stato, molte leve

`````{tab} Elementare

Dieci leve in fila, mille tiri da spendere. Ogni leva paga un punteggio diverso
a ogni tiro: qualcuna in media è generosa, qualcuna avara, e quali siano non lo
sai. Sono punti e non denaro, e possono venire negativi. Andarsene non si può:
la domanda non è *se* tirare, ma *quale* leva tirare.

Quello che vedi del mondo in un istante si chiama **stato**: in un videogioco è
la schermata e cambia a ogni mossa, qui è quella fila di dieci leve e non cambia
mai, qualunque cosa tu faccia. Un solo stato.

Ti resta il quaderno, una pagina per leva, e sopra una riga sola: la media di
quanto quella leva ha reso finora, la tua **stima**. Dopo due tiri non vale
niente, a furia di tirare migliora. A ogni tiro la sposti invece di rifare la
somma da capo: quanto hai appena incassato meno quanto c'era scritto è la
sorpresa, e tu ne correggi una frazione, scelta da te fra zero e uno, che si
chiama **passo**.

> stima nuova = stima vecchia + passo × (quello che ho appena visto − stima vecchia)

Passo a zero, la matita non si muove mai; passo a uno, cancelli tutto e riscrivi
l'ultimo tiro. Quella riga la ritroverai in ogni metodo che impara qualcosa.

Terza leva, pagina che parte da $0$, due incassi: $4$ e poi $6$. Accorcia il
passo a ogni tiro, cioè "uno diviso le volte che ho tirato questa leva", e la
riga ti dà la media di tutti i tiri: passo $1/1 = 1$, scrivi
$0 + 1 \times (4 - 0) = 4$; passo $1/2$, scrivi
$4 + 0{,}5 \times (6 - 4) = 5$, appunto la media fra $4$ e $6$. Tieni invece la
matita ferma a un mezzo, stessa pagina e stessi incassi: dopo il $4$ scrivi
$0 + 0{,}5 \times (4 - 0) = 2$, dopo il $6$ scrivi
$2 + 0{,}5 \times (6 - 2) = 4$.

Quel $4$ non è la media, che sarebbe $5$. Metà viene dal $6$ appena visto
($0{,}5 \times 6 = 3$), un quarto dal $4$ di prima ($0{,}25 \times 4 = 1$), e
l'ultimo quarto è lo zero della prima riga, che occupa il posto senza portare
niente. A ogni tiro il passato sbiadisce della stessa frazione (a un mezzo il
peso di un vecchio tiro si dimezza, con un passo più corto cala più adagio), e
con lui sbiadisce quello zero, che nessuna leva ha mai pagato. Ti serve quando
le leve cambiano carattere mentre giochi, e fuori dal casinò succede sempre.

Alla lunga i due modi si separano. Col passo che si accorcia ogni tiro sposta la
riga meno del precedente, e la pagina finisce per fermarsi sul valore vero della
leva. Col passo fermo l'ultimo tiro pesa sempre uguale, e la riga balla attorno
al valore vero anche se la leva non è cambiata di una virgola. Al passo chiedi
due cose: portare la pagina lontano dallo zero di partenza, e accorciarsi tanto
in fretta da smettere di rincorrere la fortuna di un tiro solo. Uno diviso il
numero di tiri le fa tutt'e due, il passo fermo solo la prima, e per questo
insegue per sempre, che è precisamente quello che gli chiedi.

`````

`````{tab} Superiore

Il problema ha $k$ azioni. Ogni azione $a$ ha un valore vero
$q_*(a) = \mathbb{E}[R_t \mid A_t = a]$, ignoto, e la ricompensa osservata è
una realizzazione rumorosa attorno a quel valore. Non c'è stato: la
distribuzione delle ricompense non dipende da cosa è successo prima. Un bandit
è, se si vuole, un MDP con un solo stato.

Un avvertimento sugli indici, perché è una trappola classica e cade proprio
qui. Dove lo stato manca, la ricompensa dell'azione $A_t$ si indicizza $R_t$,
perché non c'è uno stato successivo di cui tenere il passo; dalla prossima
sezione in poi, dove lo stato c'è, la stessa ricompensa si scriverà $R_{t+1}$,
come nella panoramica. È la convenzione di Sutton e Barto ed è comoda da
entrambe le parti, ma va tenuta a mente confrontando le formule di qui con
quelle di là.

La stima naturale di $q_*(a)$ è la **media campionaria**

$$
Q_t(a) = \frac{\sum_{i<t} R_i \cdot \mathbb{1}[A_i = a]}{\sum_{i<t}
\mathbb{1}[A_i = a]},
$$

dove $\mathbb{1}[\cdot]$ vale $1$ quando la condizione è vera e $0$ altrimenti:
al numeratore somma le sole ricompense incassate tirando $a$, al denominatore
conta quante volte $a$ è stata tirata.

Si calcola in forma incrementale, senza tenere in memoria la storia. Se
$Q_n$ è la stima dopo $n-1$ tiri della stessa leva e $R_n$ è l’$n$-esima
ricompensa,

$$
Q_{n+1} = Q_n + \frac{1}{n}\big(R_n - Q_n\big).
$$

È la forma canonica di tutte le regole di apprendimento che seguiranno:
*stima $\leftarrow$ stima $+$ passo $\cdot$ errore*. La ritroveremo identica
nel TD, dove l'errore è l'errore temporale, e imparentata nella discesa del
gradiente, dove il passo è il learning rate.

Sostituendo $1/n$ con un passo costante $\alpha \in (0,1]$ si ottiene invece

$$
Q_{n+1} = (1-\alpha)^n Q_1 + \sum_{i=1}^{n} \alpha (1-\alpha)^{n-i} R_i ,
$$

una **media pesata esponenzialmente sul recente**: i pesi decadono
geometricamente all'indietro. Non converge (continua a inseguire), ed è
esattamente ciò che serve quando il problema è **non stazionario**, cioè quando
$q_*(a)$ cambia nel tempo. Il caso stazionario è l'eccezione, non la regola.

Il criterio che distingue i due casi è lo stesso che tornerà a chiedere la
garanzia di convergenza del Q-learning. Una successione di passi $\alpha_n$
porta la stima al valore vero se soddisfa le **condizioni di Robbins-Monro**

$$
\sum_{n=1}^{\infty} \alpha_n = \infty,
\qquad
\sum_{n=1}^{\infty} \alpha_n^2 < \infty :
$$

la prima chiede che i passi restino abbastanza grandi da lasciarsi alle spalle
qualunque punto di partenza, la seconda che si accorcino abbastanza in fretta da
smettere di rincorrere il rumore. Il passo $1/n$ le soddisfa entrambe
($\sum 1/n$ diverge, $\sum 1/n^2$ converge); un passo costante $\alpha$ soddisfa
la prima e viola la seconda, ed è per questo che non converge. È la proprietà
che lo rende adatto ai problemi non stazionari.

`````

## Il costo di essere avidi

Un agente **avido** (*greedy*) tira sempre la leva con la stima più alta. Il
guaio è che la stima più alta all'inizio è quasi sempre sbagliata, e l'errore
si auto-conferma: se la leva davvero migliore ha avuto sfortuna nei primi due
tiri, la sua stima resta bassa, non viene più scelta, e nessuno la corregge
mai. L'agente si chiude dentro una convinzione senza aver mai raccolto le prove
per smentirla.

Quanto costi si misura su un banco di prova, sempre lo stesso: è quello di
Sutton e Barto {cite}`sutton2018reinforcement`, i due autori del manuale
classico della materia. Il banco di prova è il loro, ed è sempre lo stesso.

`````{tab} Elementare

Il banco è fatto così. Dieci leve, mille tiri in tutto. Ogni leva ha un suo
**valore vero**, cioè quanto rende in media. Quei dieci valori li sorteggia,
prima di cominciare, chi ha costruito l'esperimento, e li sorteggia attorno
allo zero: qualche leva rende un po’ più di zero, qualcuna un po’ meno,
nessuna moltissimo. Quando tiri una leva incassi il suo valore vero più un
errore casuale, che di solito sta fra il meno uno e il più uno: ecco perché
due tiri della stessa leva danno numeri diversi, ed ecco perché le tue stime,
all'inizio, non valgono niente. E poiché con dieci leve sorteggiate una volta
sola si rischia di essere fortunati o sfortunati per caso, l'esperimento intero
si rifà da capo duemila volte con leve nuove e si fa la media dei risultati.

Il punteggio con cui si giudica una strategia è la percentuale di volte in cui,
**negli ultimi cento tiri**, sta tirando davvero la leva migliore. Qui c'è una
cosa da dire, perché sembra una contraddizione e non lo è: qual è la leva
migliore l'agente non lo sa e non lo saprà mai, ma chi ha costruito
l'esperimento sì, perché quei valori li ha sorteggiati lui. È il vantaggio di
un banco di prova su una vera macchinetta da casinò, e serve esattamente a
questo: dare un voto. Ultimi cento tiri e non tutti e mille, perché all'inizio
sbagliare è inevitabile: quel che interessa è che cosa ha imparato alla fine.

Quel voto, però, dice dove sei arrivato e non quanto ti è costato arrivarci. Due
strategie che negli ultimi cento tiri azzeccano la leva migliore altrettanto
spesso possono aver buttato per strada quantità di punti molto diverse, e il
voto le mette pari lo stesso.

`````

`````{tab} Superiore

Il banco di prova è il *10-armed testbed*: dieci leve i cui valori veri
$q_*(a)$ sono estratti da una normale standard $\mathcal{N}(0,1)$, ricompense
$R_t \sim \mathcal{N}(q_*(A_t), 1)$, mille tiri, e tutto ripetuto su duemila
banchi indipendenti per mediare la fortuna dell'estrazione. La misura riportata
è la frequenza di scelta dell'azione ottima negli ultimi cento tiri, che è una
misura del comportamento **asintotico** rispetto a quell'orizzonte, non della
ricompensa accumulata lungo la strada: due strategie con la stessa frequenza
finale possono aver pagato prezzi molto diversi per arrivarci.

`````

Su quel banco, negli ultimi cento tiri e in media su tutti e duemila gli
esperimenti, l'agente avido sceglie la leva migliore solo nel **36,7%** dei
casi: dopo mille tentativi, due volte su tre sta ancora tirando la leva
sbagliata.

Basta pochissimo per cambiare le cose. Con $\varepsilon$-greedy, cioè una leva
a caso una volta ogni dieci, si sale all’**80,2%**. Un numero da scegliere però
c'è: $\varepsilon$ lo si sceglie, e poche
pagine più avanti si vedrà che va scelto guardando quanto dura la partita. Ha
comunque una virtù: sbagliarlo per difetto costa poco. Azzardare una volta su
cento invece che una su dieci, cioè dieci volte di meno, sullo stesso banco dà
il **59,1%**: parecchio sotto l'80,2%, ma parecchio sopra il 36,7% di chi non
azzarda mai. Sbagliarlo per eccesso invece costa carissimo, e il motivo
si vede a occhio: chi azzarda a ogni tiro non sfrutta mai quello che ha
imparato, tira sempre a caso, e con dieci leve indovina una volta su dieci
esattamente come al primo tiro. Il difetto vero, però, è un altro, ed è
altrettanto chiaro: quando esplora, esplora **a casaccio**. Tira con la stessa
probabilità la leva che potrebbe essere la seconda migliore e quella che ha già
dimostrato dieci volte di essere pessima. Le tre idee che seguono spendono
l'esplorazione meglio.

## Tre modi di esplorare meglio di un dado

### Valori iniziali ottimisti

L'idea più economica non tocca l'algoritmo: cambia soltanto il numero da cui
partono le stime, e in cambio chiede di tenere il passo fisso invece di fare la
media.

`````{tab} Elementare

Sul banco di prova le leve rendono, in media, attorno a zero. Scriviamo allora
sul quaderno, prima ancora di aver tirato, che ogni leva vale $+5$: una promessa
che nessuna leva può mantenere. Qualunque leva si tiri, quello che si incassa è
meno di quanto c'era scritto, quindi la sua stima scende sotto quelle delle leve
non ancora provate, e al tiro dopo l'agente ne prova un'altra. Un agente che
sceglie sempre la migliore secondo il quaderno finisce così per girarle tutte,
senza tirare nessun dado: esplora perché resta sistematicamente deluso.

Perché la delusione duri, però, il passo deve restare **fisso** (nell'esperimento
più avanti vale un decimo). Se il passo è invece "uno diviso il numero di
tiri", cioè se la stima è la media di tutti i tiri, il primo tiro se lo porta
via da solo:
la stima salta di colpo sul numero appena visto, e su quella leva l'ottimismo è
finito. Resta il giro forzato sulle altre nove, e infatti anche così si arriva
al **71,3%**, quasi il doppio del 36,7% dell'agente avido. Una parte del
guadagno però se n'è andata, e conviene misurarla invece di dirla a occhio. Con
il passo fisso si arriva all’**86,6%**, quasi cinquanta punti sopra l'avido
($86{,}6 - 36{,}7 = 49{,}9$), e su mille tiri nessun'altra farà meglio; con la
media se ne guadagnano trentaquattro e mezzo ($71{,}3 - 36{,}7 = 34{,}6$). Quindici
punti su cinquanta sono rimasti sul tavolo, quasi un terzo.

`````

`````{tab} Superiore

Sul banco di prova $q_*(a) \sim \mathcal{N}(0,1)$. Inizializzando $Q_1(a) = +5$
per ogni $a$, qualunque azione si scelga la ricompensa osservata **delude** in
media di cinque unità, la stima corrispondente scende sotto quelle delle azioni
non ancora provate, e l'agente, pur restando strettamente greedy, spazza
l'intero insieme delle azioni.

Perché l'ottimismo sopravviva al primo tiro serve però un passo costante. Con la
media campionaria $Q_2 = Q_1 + \frac{1}{1}(R_1 - Q_1) = R_1$: la stima **è** la
prima ricompensa osservata, e l'ottimismo su quella leva evapora in un colpo
solo. Resta l'ottimismo sulle altre nove, che impone comunque un giro completo,
e infatti si arriva al **71,3%** contro il **36,7%** dell'avido puro: circa il
69% del divario fra l'avido e l’86,6% che lo stesso ottimismo raggiunge con il
passo costante. Il termine di paragone corretto per l'ottimismo è l'avido, non
$\varepsilon$-greedy; resta però che con la media campionaria il risultato
finisce sotto l'80,2% della leva a
caso una volta ogni dieci, mentre con $\alpha = 0{,}1$ costante l'ottimismo si
consuma abbastanza lentamente da arrivare all’**86,6%**.

`````

È un trucco, però, e conviene dire perché. L'ottimismo si esaurisce: dopo che
tutte le leve sono state provate abbastanza, la spinta a esplorare sparisce.
Se le leve rendono sempre allo stesso modo (si dice che il problema è
**stazionario**) va benissimo; se invece cambiano carattere nel tempo, e
servirebbe tornare a esplorare perché quel che si era imparato non vale più,
non serve a niente. Come scrivono Sutton e Barto, l'inizio del tempo capita una
volta sola, e non conviene puntarci troppo.

### UCB: esplorare in proporzione a quanto poco si sa

Il numero su cui si decide, qui, è il valore più alto che quella leva potrebbe
ancora avere viste le prove raccolte finora, e non la sua stima. Da
lì il nome, che sono tre lettere per *upper confidence bound*, alla lettera «il
tetto di quello che ancora ci si può ragionevolmente aspettare».

`````{tab} Elementare

Il difetto di $\varepsilon$-greedy è che il dado non guarda in faccia nessuno.
Ma fra le leve non scelte ce ne sono di due tipi diversissimi: quelle che
abbiamo provato venti volte e sono chiaramente mediocri, e quelle che abbiamo
provato una volta sola, per cui non sappiamo davvero niente. Le prime non
meritano un altro tiro, le seconde sì.

L'idea è aggiungere alla stima di ogni leva un **bonus di ignoranza**: quanto
più raramente l'ho tirata, tanto più generoso è il bonus. Poi si sceglie, senza
dadi, la leva con la somma più alta. Una leva mediocre ma poco esplorata può
vincere il confronto proprio grazie al bonus; ogni volta che la si tira il
bonus cala, finché la sua mediocrità non emerge e smette di essere scelta.

C'è un ultimo pezzo, ed è quello che sorprende: il bonus di una leva cresce anche
per il solo passare del tempo, cioè anche nei tiri in cui quella leva non la si
tira. Il motivo è che l'ignoranza si misura per confronto. Se in cento tiri ho
provato una leva due volte, di lei so poco; se in mille tiri l'ho sempre
provata due volte, di lei so poco esattamente come prima, ma di tutte le altre
adesso so molto di più, e la sproporzione è cresciuta. Il risultato è che una
leva trascurata a lungo torna prima o poi in cima alla lista: nessuna viene
abbandonata per sempre, ma le peggiori vengono ricontrollate sempre più di
rado.

Sul solito banco di prova UCB azzecca la leva migliore l’**85,9%** delle volte:
praticamente quanto l'ottimismo iniziale, e nettamente meglio della leva a caso
una volta ogni dieci.

Quel punteggio lascia fuori il costo del viaggio. A fine partita, confronta
quello che hai incassato con quello che avresti incassato sapendo dalla prima
mossa qual è la leva migliore: la differenza è il **rimpianto**. A zero non può
andare, perché per sapere che una leva è scarsa bisogna tirarla, e per esserne
sicuri bisogna tirarla più di una volta.

Il rimpianto di UCB cresce sempre più adagio: i primi cento tiri se ne portano
via un bel pezzo, i secondi cento molto meno, e più la partita va avanti più di
rado si sbaglia. Chi tira a caso una volta ogni dieci paga invece la stessa
cifra ogni cento tiri per sempre, e il suo conto cresce in linea retta; su una
partita lunga la distanza fra i due diventa enorme. Il fatto notevole è che più
adagio di così non si può andare: nessuna strategia che debba funzionare su
qualunque fila di macchinette paga meno. Chi promette di pagare meno ha deciso
in anticipo, e tirare sempre la terza leva costa zero quando la terza è la
migliore e una fortuna in tutti gli altri casi.

Tutto questo però regge finché ha senso tenere il conto dei tiri di ogni leva.
Se le leve cambiano carattere mentre giochi, i conti accumulati dall'inizio
parlano di un mondo che non c'è più. E se al posto di dieci leve ci fossero
milioni di situazioni tutte diverse, la domanda «quante volte ho già provato
questa» resterebbe senza risposta: nessuna capiterebbe due volte, il bonus
verrebbe identico per tutte, e non distinguerebbe più niente.

`````

`````{tab} Superiore

L’**Upper Confidence Bound** sceglie

$$
A_t = \arg\max_{a} \left[\, Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \,\right],
$$

dove $N_t(a)$ è il numero di volte che $a$ è stata scelta prima di $t$ e
$c > 0$ regola quanto pesa l'incertezza (le azioni mai provate si trattano come
massimamente urgenti). Il termine sotto radice è, a meno di costanti, la
larghezza di un intervallo di confidenza sulla media di $a$: il numeratore
$\ln t$ cresce con il tempo, il denominatore $N_t(a)$ con l'uso. La forma
$\sqrt{\ln t / N_t(a)}$ non è arbitraria: esce da una disuguaglianza di
concentrazione (Hoeffding, che per ricompense in $[0,1]$ dà
$\Pr(|\bar{X}_n - \mu| \ge u) \le 2e^{-2nu^2}$) applicata alla media di
$N_t(a)$ campioni, con il $\ln t$ che paga l'unione su tutti i passi fatti
finora. Il nome dice il principio: **ottimismo di fronte all'incertezza**, cioè
agire come se ogni azione valesse il massimo compatibile con i dati raccolti, e
lasciare che siano i dati a smentire.

Il decadimento logaritmico non è decorativo. Lai e Robbins
{cite}`lai1985asymptotically` dimostrano che nessun algoritmo buono su *tutte*
le istanze del problema (con rimpianto sub-polinomiale qualunque siano i
valori delle leve: la clausola esclude scorciatoie come tirare sempre la
stessa leva, che trionfa quando quella leva è la migliore e affonda su tutte
le altre istanze) può avere un **rimpianto**

$$
\mathcal{R}_T = T \max_a q_*(a) - \mathbb{E}\!\left[\sum_{t=1}^{T} R_t\right]
$$

che cresca, asintoticamente, meno che logaritmicamente in $T$: perdere
qualcosa è inevitabile, la domanda è solo quanto. Auer, Cesa-Bianchi e
Fischer {cite}`auer2002finite` mostrano che UCB1 raggiunge quella crescita
logaritmica con una garanzia valida a ogni istante finito, non solo
asintoticamente; resta però sopra la costante ottima di Lai e Robbins (la
raggiungono varianti più fini, come KL-UCB), e il teorema assume ricompense
limitate, un'ipotesi che il banco di prova gaussiano a dieci leve, a rigore, non
rispetta.

Per confronto, $\varepsilon$-greedy con $\varepsilon$ costante ha rimpianto
**lineare** in $T$, perché continua a sbagliare una frazione fissa delle volte
per sempre: è la differenza fra un'esplorazione che si dosa e una che non si
spegne mai.

Sul banco di prova, con $c = 2$, UCB sceglie la leva migliore l’**85,9%** delle
volte. Attenzione a non leggere quel $c=2$ come la costante del teorema: è la
scelta empirica di Sutton e Barto, mentre UCB1 nella forma dimostrata da Auer e
colleghi corrisponde a $c = \sqrt{2}$ per ricompense in $[0,1]$. Il limite
pratico è che la formula presuppone un problema stazionario e
un numero maneggiabile di azioni: portarla di peso nel reinforcement learning
con approssimazione di funzione, dove "quante volte ho visto questo stato" non
è nemmeno ben definito, non funziona, ed è il motivo per cui la sezione
sull'esplorazione nel deep RL dovrà inventarsi altro.

`````

### Il bandit a gradiente: preferenze, non valori

L'ultima strategia della sezione butta via il quaderno delle stime. Per ogni
leva tiene un voto, e dopo ogni tiro lo sposta un poco nel verso che le sembra
faccia incassare di più. Il **gradiente** del titolo è quello del capitolo di
matematica, la direzione lungo cui una quantità cresce più in fretta: qui la
quantità da far crescere è quanto ci si aspetta di incassare, e i voti sono le
manopole da girare.

`````{tab} Elementare

Le strategie viste finora stimano *quanto vale* ogni leva e poi decidono. Se ne
può fare a meno: si può imparare direttamente una **preferenza**, cioè un voto
che non è una previsione di guadagno e non ha unità di misura, e poi tirare
ogni leva tanto più spesso quanto più alto è il suo voto.

Dai voti alle probabilità si passa così: si confrontano i dieci voti fra loro,
e ciascuna leva viene tirata tanto più spesso quanto più il suo voto **supera**
gli altri. Non conta il posto in classifica, contano i distacchi. Con dieci voti
tutti uguali si tira del tutto a caso, una leva su dieci. Se una leva prende un
punto di vantaggio su tutte le altre, viene tirata quasi una volta su quattro.
Con due punti di vantaggio siamo già a quasi una volta su due, e con tre a più
di due volte su tre: pochi punti di distacco bastano a prendersi quasi tutti i
tiri.

Ne viene che alzare tutti i voti della stessa quantità non cambia niente. È come
in una classifica a punti: se do un punto in più a tutti, l'ordine e i distacchi
restano identici, e ogni leva continua a essere tirata quanto prima.

La regola di aggiornamento è di buon senso: se la ricompensa appena incassata è
**migliore della media** di quelle ricevute finora, alzo il voto della leva che
ho tirato e abbasso quello di tutte le altre; se è peggiore, faccio l'opposto.

Di quanto lo alzo dipende da quanto quella leva veniva già tirata: se era già la
favorita il ritocco è minimo, se era una che non si tirava quasi mai è grande. E
quello che quella leva guadagna lo perdono esattamente le altre, spartito fra
loro in proporzione a quanto venivano tirate, sicché il totale dei dieci voti
resta sempre lo stesso.

Quel confronto con la media è il pezzo importante e si chiama **termine di
riferimento** (in inglese *baseline*, ed è il nome che si legge nel codice).
Senza, l'algoritmo confronta la ricompensa con lo zero, che è un numero
arbitrario, e succede questo: se tutte le leve pagano attorno a mille, ogni
tiro sembra un successo, e il voto della leva appena tirata sale forte
qualunque cosa quella leva valga davvero. A decidere non è più quale leva rende
di più, ma quale capita di tirare, e siccome una leva che sale viene tirata più
spesso, il primo colpo di fortuna si autoalimenta. La differenza vera fra le
leve c'è ancora, ma è di qualche unità sopra un mille, e resta sepolta. Con il
riferimento, quel che conta non è quanto ho preso ma quanto ho preso **rispetto
al solito**, e quel mille sparisce dal conto.

Quanto conti si misura, ed è tanto. Sul banco di prova di prima il metodo dei
voti azzecca la leva migliore l’**84,1%** delle volte. Aggiungiamo adesso
quattro punti a tutte le ricompense: quattro tanto per dire, serve solo un
numero abbastanza grande da coprire le differenze fra le leve, che su questo
banco sono dell'ordine dell'uno. Il problema non cambia in nulla, perché fra le
leve i distacchi restano quelli. Con il riferimento si resta all’**83,8%**;
senza, si crolla al **48,5%**.

`````

`````{tab} Superiore

Si mantiene una preferenza $H_t(a) \in \mathbb{R}$ per ogni azione, e la policy
è una softmax sulle preferenze:

$$
\pi_t(a) = \Pr\{A_t = a\} = \frac{e^{H_t(a)}}{\sum_{b} e^{H_t(b)}} .
$$

Le preferenze non stimano nulla, contano solo le loro differenze (aggiungerne
mille a tutte non cambia la policy). L'aggiornamento è una salita stocastica
sul gradiente della ricompensa attesa:

$$
H_{t+1}(A_t) = H_t(A_t) + \alpha\,\big(R_t - \bar{R}_t\big)\big(1 - \pi_t(A_t)\big),
\qquad
H_{t+1}(a) = H_t(a) - \alpha\,\big(R_t - \bar{R}_t\big)\,\pi_t(a)
\;\; \forall a \neq A_t ,
$$

dove $\bar{R}_t$ è la media delle ricompense incassate **prima** di $t$, cioè
la **baseline** (è quello che fa anche il codice più sotto, che aggiorna la
media dopo averla usata).

Quello che si sta guardando ha già un nome: è **REINFORCE con baseline**, il
metodo a gradiente di policy del capitolo sul deep reinforcement learning, nel
caso degenere di un solo stato. La stessa
struttura (una distribuzione parametrica sulle azioni, un aggiornamento
proporzionale alla ricompensa scostata da un riferimento) che là si scriverà
come $\nabla_\theta \log \pi_\theta(a\mid s)\,\hat{A}_t$, con il vantaggio
$\hat{A}_t$ al posto di $R_t - \bar{R}_t$ (il cappello lo mettiamo qui per non
confondere il vantaggio con l'azione $A_t$ delle formule precedenti; nel
capitolo di deep RL, dove l'ambiguità non c'è, si scriverà $A_t$). Il
*vantaggio* dell'actor-critic nasce qui, e nasce
per la stessa ragione: ridurre la varianza senza spostare la media del
gradiente.

Che la baseline serva davvero si misura. Sul banco di prova centrato in zero il
metodo arriva all’**84,1%**. Traslando **tutte** le ricompense di $+4$, cosa
che non cambia in nulla la difficoltà del problema (le differenze fra le leve
sono identiche), con la baseline si resta all’**83,8%**, mentre togliendola si
crolla al **48,5%**. La baseline non è un'ottimizzazione: è ciò che rende
l'algoritmo indifferente all'origine della scala delle ricompense.

`````

## Alla prova: duemila banchi da mille tiri

Le prime quattro strategie (l'avida, quella che azzarda ogni tanto,
l'ottimista e UCB) stanno in un blocco solo, perché fra loro cambiano in due
punti soltanto: come scelgono la leva e da quale numero partono le stime. Le
righe stampate sono sei e non quattro perché chi azzarda compare due volte, con
due frequenze diverse, e l'ottimista anche, con i due modi di fare il passo.

```python
import numpy as np

K, PASSI, PROVE = 10, 1000, 2000     # 10 leve, 1000 tiri, 2000 banchi di prova

def prova(eps, q0=0.0, c=None, alpha=None):
    rng = np.random.default_rng(20260807)
    q_vero = rng.normal(0, 1, size=(PROVE, K))   # il valore vero di ogni leva
    ottima, righe = q_vero.argmax(axis=1), np.arange(PROVE)
    Q = np.full((PROVE, K), q0, dtype=float)     # le nostre stime
    N = np.zeros((PROVE, K))                     # quante volte ho tirato ogni leva
    centri = np.zeros(PASSI)
    for t in range(1, PASSI + 1):
        if c is None:
            a = Q.argmax(axis=1)
            caso = rng.random(PROVE) < eps       # ogni tanto, una leva a caso
            a = np.where(caso, rng.integers(0, K, PROVE), a)
        else:                                    # UCB: stima + incertezza
            bonus = np.where(N == 0, 1e6, c * np.sqrt(np.log(t) / np.maximum(N, 1e-9)))
            a = (Q + bonus).argmax(axis=1)
        r = rng.normal(q_vero[righe, a], 1.0)    # la ricompensa e' rumorosa
        N[righe, a] += 1
        passo = alpha if alpha else 1.0 / N[righe, a]   # media incrementale
        Q[righe, a] += passo * (r - Q[righe, a])        # vecchia + passo * errore
        centri[t-1] = (a == ottima).mean()
    return 100 * centri[-100:].mean()

print(f"greedy                    {prova(eps=0.0):5.1f}%")
print(f"eps-greedy 0,01           {prova(eps=0.01):5.1f}%")
print(f"eps-greedy 0,1            {prova(eps=0.1):5.1f}%")
print(f"ottimista Q1=5, passo 0,1 {prova(eps=0.0, q0=5.0, alpha=0.1):5.1f}%")
print(f"ottimista Q1=5, media     {prova(eps=0.0, q0=5.0):5.1f}%")
print(f"UCB c=2                   {prova(eps=0.0, c=2.0):5.1f}%")

# greedy                     36.7%
# eps-greedy 0,01            59.1%
# eps-greedy 0,1             80.2%
# ottimista Q1=5, passo 0,1  86.6%
# ottimista Q1=5, media      71.3%
# UCB c=2                    85.9%
```

Un risultato merita attenzione, ed è quello di chi esplora una volta su cento:
dopo mille tiri sta al **59,1%**, contro l’**80,2%** di chi esplora una volta su
dieci, e sembra il peggiore dei rimedi. Ma sta ancora salendo. Esplorando una
volta su cento impiega dieci volte più tempo a farsi un'idea di tutte le leve, e
alla fine supera l'altro, che invece continuerà per sempre a buttare un tiro su
dieci. Portando `PASSI` da mille a trentamila, cioè cambiando una
riga sola del programma, il sorpasso arriva passato il novemillesimo tiro, e
alla fine chi azzarda una volta su cento sta al **91,6%** e chi azzarda una
volta su dieci all’**89,0%**: le parti si sono invertite. La classifica dipende insomma
da quanto è lunga la partita, e questa è una morale generale:
**quanto esplorare
si decide guardando l'orizzonte**, cioè il numero di tiri che si hanno davanti,
non i primi mille.

Il bandit a gradiente ha una struttura diversa e sta in un blocco a sé, dove si
vede anche l'esperimento sulla baseline.

```python
import numpy as np

K, PASSI, PROVE = 10, 1000, 2000

def gradiente(alpha=0.1, baseline=True, shift=0.0):
    rng = np.random.default_rng(20260807)
    q_vero = rng.normal(shift, 1, size=(PROVE, K))
    ottima, righe = q_vero.argmax(axis=1), np.arange(PROVE)
    H = np.zeros((PROVE, K))          # preferenze: non sono valori, sono voti
    media_r, centri = np.zeros(PROVE), np.zeros(PASSI)
    for t in range(1, PASSI + 1):
        p = np.exp(H - H.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)                    # softmax
        a = (p.cumsum(axis=1) < rng.random((PROVE, 1))).sum(axis=1).clip(0, K-1)
        r = rng.normal(q_vero[righe, a], 1.0)
        scelta = np.zeros((PROVE, K)); scelta[righe, a] = 1.0
        base = media_r if baseline else 0.0                  # il termine di confronto
        H += alpha * (r - base)[:, None] * (scelta - p)      # sali sul gradiente
        media_r += (r - media_r) / t
        centri[t-1] = (a == ottima).mean()
    return 100 * centri[-100:].mean()

print(f"gradiente, ricompense centrate su 0   {gradiente():5.1f}%")
print(f"gradiente, ricompense centrate su +4  {gradiente(shift=4.0):5.1f}%")
print(f"  ... senza baseline                  {gradiente(shift=4.0, baseline=False):5.1f}%")

# gradiente, ricompense centrate su 0    84.1%
# gradiente, ricompense centrate su +4   83.8%
#   ... senza baseline                   48.5%
```

## Tirare a sorte da quello che si crede: il Thompson sampling

Le strategie messe alla prova finora riassumono ogni leva in **una stima
sola**: quanto si crede che renda, con eventualmente un bonus accanto o un voto
al suo posto. Resta la risposta che Thompson diede nel 1933, e che cambia
proprio la cosa riassunta.

`````{tab} Elementare

Di ogni leva si tiene tutto quello che di quella leva è ancora possibile
credere. Di una mai tirata si può credere quasi tutto: potrebbe essere ottima o
pessima, e il ventaglio delle possibilità è largo. Dopo dieci tiri si è spostato
attorno a quello che si è visto e si è ristretto; dopo mille è quasi un punto.

Il ventaglio si muove a ogni tiro, sempre nello stesso modo: il centro scivola
verso l'incasso appena visto, i bordi si avvicinano. Quanto scivola dipende da
quanti tiri pesano già dentro, e il ventaglio di partenza conta come **tiri
finti**: uno largo come il ventaglio giusto per questo banco ne pesa uno, e
allora il primo tiro vero sposta il centro di mezzo passo, il decimo di un
undicesimo, il millesimo quasi di niente.

La mossa sta qui: prima di ogni tiro, dentro ogni ventaglio si **sorteggia un
valore**, come si scommettesse su una delle versioni del mondo ancora possibili;
poi si tira la leva che in quel sorteggio è uscita più alta. Non tutti i punti
del ventaglio sono ugualmente probabili: quelli vicino al centro escono spesso,
quelli sui bordi di rado.

L'esplorazione arriva da sola, senza che nessuno debba deciderla. Una leva di
cui si sa poco ha il ventaglio largo, quindi ogni tanto le esce un valore alto e
viene tirata; una leva provata cento volte e mediocre ha il ventaglio stretto
attorno a un numero basso, e un valore alto non le esce quasi più. Non c'è
nessun dado da regolare come in $\varepsilon$-greedy né nessun bonus da
scegliere a mano come in UCB: la larghezza del ventaglio fa il lavoro, e si
stringe da sé mentre le prove si accumulano.

Ed è il dado truccato del 1933, guardato da vicino. Una leva finisce per essere
tirata tante volte quante sono quelle in cui, fra tutti i sorteggi, tocca a lei
uscire più alta; e quel numero è, per come è fatto il sorteggio, quanto è
probabile che sia lei la migliore secondo ciò che si crede in quel momento. Le
due frasi dicono la stessa cosa. La cura del paziente che sta perdendo non viene
mai sospesa del tutto, e la sua quota cala nella misura esatta in cui le prove
la smentiscono.

Sul solito banco di prova la leva migliore viene azzeccata il **91,8%** delle
volte, contro l’**85,9%** di UCB e l’**86,6%** dell'ottimismo iniziale. E il
conto di quello che si perde per strada cresce con il ritmo più lento possibile,
lo stesso di UCB, cioè sempre più adagio; ma a parità di ritmo si può pagare di
più o di meno, e qui sta la differenza. Il minimo sotto cui nessuna strategia
può scendere, quello di cui parlava la sezione su UCB, UCB stesso lo avvicina
senza toccarlo; il sorteggio di Thompson lo tocca. Dimostrato, però, per il caso
in cui ogni tiro finisce con un sì o con un no, che è quello della
sperimentazione clinica da cui tutto è partito. Sul banco delle dieci leve, dove
gli incassi sono numeri qualsiasi, il 91,8% resta una misura e non
l'illustrazione di un teorema.

Il prezzo va detto, perché non è zero. Per sorteggiare dentro un ventaglio
bisogna prima dire quanto è largo quello di partenza, prima di aver visto
niente, e bisogna sapere quanto è ballerino l'incasso di un tiro singolo (qui lo
si sa, perché è il banco di prova a dichiararlo). Partire vaghi non costa quasi
nulla: un ventaglio cinque volte più largo del necessario porta al **91,6%**,
cioè allo stesso posto, perché pesa meno di un tiro finto e il primo tiro vero
se lo porta via. Partire convinti costa: un ventaglio dieci volte più stretto
del giusto pesa **cento** tiri finti, e quei cento tiri finti hanno pagato tutti
**zero**, perché è lì che il ventaglio era centrato. Per convincerlo che una
leva rende davvero ne servono cento veri. Si scende al **75,2%**, cioè si
buttano più di sedici punti per una convinzione che nessuno aveva verificato.

Ed è il ventaglio **stretto** a fare danno, non quello spostato in alto: partire
convinti che tutte le leve siano ottime è il trucco dei valori iniziali
ottimisti di qualche pagina fa, e quello funziona. Resta poi il limite di tutta
la sezione: se le leve cambiano carattere mentre si gioca, un ventaglio già
stretto attorno al mondo di prima non si riapre da sé, e bisogna fargli
dimenticare le prove vecchie.

`````

`````{tab} Superiore

Si mette una distribuzione a priori su ciascun valore $q_*(a)$ e la si aggiorna
con le ricompense osservate, ottenendo a ogni istante $t$ una posteriore
$P^{(a)}_t$, cioè quello che si crede di $q_*(a)$ viste le prove raccolte fino a
$t$. La regola di scelta è

$$
A_t = \arg\max_{a} \; \tilde{q}_t(a),
\qquad \tilde{q}_t(a) \sim P^{(a)}_t
\;\; \text{indipendentemente per ogni } a,
$$

dove $\tilde{q}_t(a)$ è un singolo campione estratto dalla posteriore del
braccio $a$ (la letteratura scrive $\theta_a$; qui no, perché $\theta$ in
questo capitolo sono i parametri di una policy). Un campione per braccio, il
massimo vince. La probabilità che così esca $a$ è la probabilità a posteriori
che $a$ sia il braccio ottimo, e per questo il metodo si chiama anche
*probability matching*: la frequenza con cui si gioca un'azione insegue la
probabilità che sia la migliore. L'uguaglianza vuole però un'ipotesi, e conviene
dirla perché è quella che cade per prima: la posteriore **congiunta deve
fattorizzarsi** sui bracci, cosa che qui vale perché il priore è indipendente e
i tiri di un braccio non dicono niente sugli altri. Con bracci correlati, che è
il caso del bandit contestuale nominato in coda alla sezione, sorteggiare
braccio per braccio smette di coincidere con la probabilità di essere il
migliore.

Sul banco di prova gaussiano le ricompense sono
$R_t \sim \mathcal{N}(q_*(a), \sigma^2)$ con $\sigma^2 = 1$ nota, e la coniugata
è di nuovo una gaussiana. Con priore $\mathcal{N}(0, \sigma_0^2)$, e con
$N_t(a)$ e $Q_t(a)$ i soliti conteggio e media campionaria della sezione, la
posteriore del braccio $a$ è

$$
\mathcal{N}\!\left(
\frac{N_t(a)\,Q_t(a)/\sigma^2}{1/\sigma_0^2 + N_t(a)/\sigma^2},\;
\frac{1}{1/\sigma_0^2 + N_t(a)/\sigma^2}\right),
$$

dove $1/\sigma_0^2 + N_t(a)/\sigma^2$ è la **precisione**, l'inverso della
varianza.
Letta così, la formula dice una cosa sola: il priore vale
$\sigma^2/\sigma_0^2$ osservazioni fittizie, e ogni tiro vero ne aggiunge una.
Con $\sigma = 1$ e $\sigma_0 = 0{,}1$ quelle fittizie sono cento, e valgono
tutte quanto la media del priore: è la ragione per cui un priore stretto
paralizza l'aggiornamento e lo inchioda dove era centrato. Per ricompense in
$\{0,1\}$ la coniugata è la Beta: con priore uniforme, dopo $s$ successi e $f$
fallimenti la posteriore è $\mathrm{Beta}(1+s,\, 1+f)$, ed è il caso della
sperimentazione clinica del 1933 e del test A/B.

Il meccanismo che spegne l'esplorazione è la varianza a posteriori, che va a
zero come $1/N_t(a)$: i campioni si concentrano sulla media e le azioni
chiaramente peggiori smettono di vincere il massimo. Attenzione a leggere
$N_t(a)$ e non $t$: proprio i bracci la cui esplorazione deve spegnersi sono
quelli tirati $\Theta(\ln t)$ volte, quindi la loro posteriore si stringe come
$1/\ln t$, ed è quella lentezza a impedire di impegnarsi troppo presto. Non c'è
nessun iperparametro di esplorazione, né l’$\varepsilon$ né il $c$ di UCB; il
posto di quel parametro lo prende la scelta del modello.

Sulle garanzie il metodo ha aspettato ottant'anni. Chapelle e Li
{cite}`chapelle2011empirical` lo riportano in circolazione mostrando che regge
il confronto con UCB su dati veri, e osservano che fino ad allora era rimasto
sorprendentemente poco popolare in letteratura. La prima analisi con rimpianto
logaritmico in tempo finito è di Agrawal e Goyal {cite}`agrawal2012analysis`;
Kaufmann, Korda e Munos {cite}`kaufmann2012thompson` chiudono il conto per le
ricompense di Bernoulli, dimostrando che il rimpianto raggiunge
**asintoticamente la costante** del limite inferiore di Lai e Robbins, cioè
quella soglia che UCB1 avvicina senza toccare. Stesso ordine di crescita,
costante migliore. Il banco delle dieci leve è però gaussiano, e per quel caso
la garanzia sta altrove e chiede un priore diverso: il 91,8% è una misura, non
l'illustrazione di un teorema.

Il punto di rottura sta dove sta il guadagno: la garanzia riguarda il modello,
non il mondo. Un priore troppo stretto produce una posteriore che si stringe
attorno alla cosa sbagliata, e il sorteggio smette di esplorare proprio mentre
dovrebbe. Una $\sigma$ sbagliata guasta invece nei due versi, e il secondo è
quello che sorprende: crederla più piccola del vero fa pesare troppo ogni
singolo incasso e impegnare presto; crederla più grande tiene la posteriore
larga per sempre, e allora il sorteggio continua a esplorare quando non ci
sarebbe più niente da scoprire. E come tutto il resto
della sezione, presuppone un problema stazionario: se i bracci cambiano
carattere, la posteriore accumulata descrive un mondo che non c'è più, e
occorre farla dimenticare.

`````

Il codice tiene per ogni leva gli stessi due numeri di sempre, quante volte
l'ho tirata e quanto ha reso in tutto (che è la media di prima, non ancora
divisa), e da quei due ricava a ogni passo il centro e l'ampiezza del ventaglio.
Tenere un ventaglio, insomma, non costa più che tenere una stima. La
`larghezza` è l'ampiezza di partenza: quella giusta per questo banco è $1$,
perché è così che il banco sorteggia il valore vero delle leve. Ogni caso gira
con due **semi**, cioè con due sorteggi di partenza diversi del banco: una
differenza più piccola di quanto il numero balla cambiando seme non è una
differenza.

```python
import numpy as np

K, PASSI, PROVE = 10, 1000, 2000     # le stesse 10 leve e gli stessi 2000 banchi

def thompson(larghezza, seme):
    """`larghezza` è quanto è largo il ventaglio di partenza di ogni leva."""
    rng = np.random.default_rng(seme)
    q_vero = rng.normal(0, 1, size=(PROVE, K))
    ottima, righe = q_vero.argmax(axis=1), np.arange(PROVE)
    somme = np.zeros((PROVE, K))      # quanto ha reso in tutto ogni leva
    N = np.zeros((PROVE, K))          # quante volte l'ho tirata
    centri = np.zeros(PASSI)
    for t in range(PASSI):
        peso = 1 / larghezza**2 + N   # i tiri veri, più quelli finti del ventaglio
        centro, ampiezza = somme / peso, 1 / np.sqrt(peso)
        a = rng.normal(centro, ampiezza).argmax(axis=1)   # un sorteggio per leva
        r = rng.normal(q_vero[righe, a], 1.0)
        N[righe, a] += 1
        somme[righe, a] += r
        centri[t] = (a == ottima).mean()
    return 100 * centri[-100:].mean()

for nome, larghezza in [("giusto (1)", 1.0), ("vago (5)", 5.0), ("convinto (0,1)", 0.1)]:
    esiti = [thompson(larghezza, seme) for seme in (20260807, 1)]
    print(f"ventaglio {nome:15} " + "  ".join(f"{e:.1f}%" for e in esiti))
```

```text
ventaglio giusto (1)      91.8%  91.9%
ventaglio vago (5)        91.6%  91.9%
ventaglio convinto (0,1)  75.2%  74.0%
```

Le prime due righe si sovrappongono: fra il ventaglio giusto e quello cinque
volte troppo largo ci sono due decimi di punto su un seme e nessuno sull'altro,
cioè meno di quanto ciascuno dei due si sposti cambiando seme, e fra loro non
c'è niente da scegliere. La terza sta un mondo più in basso, con i suoi due
semi vicini fra loro e lontani da tutto il resto: partire da un ventaglio dieci
volte più stretto del vero costa fra i sedici e i diciotto punti, e li costa
perché quei cento tiri finti da zero ci mettono un'eternità a lasciarsi
smentire. Il sorteggio non protegge da una convinzione sbagliata: la esegue.

## Dove si incontrano davvero

Un bandit non è un giocattolo teorico. È anzi probabilmente la parte di
reinforcement learning che più spesso finisce dentro programmi che girano
davvero, tutti i giorni, e ci finisce proprio perché rinuncia a tutto il resto.

**Test A/B, e il loro superamento.** Un test A/B è la prova che si fa quando si
hanno due versioni di una pagina, di un annuncio o di un prezzo e si vuole
sapere quale rende di più: si mostra la prima a metà dei visitatori e la seconda
all'altra metà, fino alla fine dell'esperimento. È la sperimentazione clinica di
Thompson, con gli stessi costi. Chi lo fa **in modo adattivo** sposta via via i
visitatori verso la versione che sta vincendo, e il *Thompson sampling* di
inizio sezione, quello che nessuno aveva guardato per decenni, è una delle
ricette con cui lo si fa: risparmia il denaro che avrebbe
buttato sulla versione perdente, e in cambio si complica la vita quando deve
tirare le somme, perché le due versioni non sono più state mostrate allo stesso
numero di persone né nello stesso momento, e i conti statistici che si fanno di
solito presuppongono di sì.

**Esplorazione nei sistemi di raccomandazione.** Un catalogo ha continuamente
oggetti nuovi, di cui nessuno sa nulla: mostrarli è esplorare, e non mostrarli
mai garantisce che nessuno saprà mai se erano buoni. È il problema che il
{doc}`capitolo sui sistemi di raccomandazione </SistemiRaccomandazione/overview>`, più avanti nel libro, chiamerà
**partenza a freddo**. Là la domanda sarà come descrivere un oggetto di cui non
si sa niente; qui è che cosa conviene fare mentre non si sa niente.

**Scegliere le impostazioni di un modello.** Qui il collegamento è letterale.
Un modello di machine learning, prima di essere addestrato, va regolato: quanto
grande farlo, quanto in fretta farlo imparare, e così via. Sono decine di
combinazioni possibili, provarle tutte fino in fondo costerebbe giorni, e allora
si prova ciascuna un pochino e si insiste sulle più promettenti. Descritto così
è un bandit, e infatti lo è: ogni combinazione è una leva, e provarla un po’ è
un tiro. Il *successive halving* («dimezzamento successivo») e **Hyperband**,
che è costruito sopra il primo, fanno esattamente questo, e il capitolo sul
machine learning li racconta per esteso. Con una differenza
rispetto alla macchinetta del casinò, e ha pure un nome, *best-arm
identification*: qui non interessa incassare molto lungo la strada, i tiri
spesi sono solo il costo della ricerca, interessa soltanto indovinare alla fine
quale fosse la leva migliore.

Fra il bandit e il reinforcement learning pieno c'è un gradino intermedio che
copre buona parte delle applicazioni reali: il **bandit contestuale**, in cui
prima di scegliere si osserva una descrizione della situazione (chi è
l'utente, che ora è, da quale pagina arriva) e la leva migliore dipende da
quella. C'è uno stato, quindi, ma le azioni non lo influenzano: il contesto
successivo arriva dal mondo, non da ciò che abbiamo fatto.

Ed è esattamente il pezzo che manca. Quando le azioni cominciano a determinare
in quale situazione ci si troverà dopo, il problema smette di essere una
sequenza di scelte indipendenti e diventa una **catena**: una mossa fatta ora
può essere pagata o riscossa fra venti mosse, e bisogna capire a quale mossa
attribuirne il merito. Serve un'impalcatura più grande, ed è quella della
prossima sezione.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Un **bandito a più braccia** è la decisione ridotta all'osso: una fila di
  leve, nessuna situazione che cambia, e il solo dilemma fra tirare quella che
  finora ha reso di più e provarne un'altra per sapere com'è.
- Il quaderno delle stime si aggiorna sempre allo stesso modo: *stima nuova =
  stima vecchia + passo per la sorpresa*. Se il passo è uno diviso il numero di
  tiri viene fuori la media di tutti i tiri; se il passo resta fisso, contano
  di più i tiri recenti, ed è quello che serve quando le leve cambiano carattere
  nel tempo.
- Chi tira sempre la leva con la stima più alta si chiude in una convinzione
  che non ha mai verificato: sul banco di prova azzecca la leva migliore solo
  nel 36,7% dei casi. Tirare una leva a caso una volta ogni dieci porta
  all'80,2%, ma è un'esplorazione cieca, che spreca tiri sulle leve già
  bocciate.
- **Quanto** azzardare non è una costante universale: si decide guardando
  quanti tiri si hanno davanti. Chi azzarda di rado impara più lentamente ma
  spreca meno, e su una partita abbastanza lunga finisce davanti a chi azzarda
  spesso.
- Due modi di spenderla meglio: partire da stime **troppo generose**, così che
  ogni leva deluda e l'agente le giri tutte da solo (86,6%, ma il trucco si
  consuma e non serve se il mondo cambia); oppure aggiungere a ogni stima un
  **bonus di ignoranza**, tanto più grande quanto meno si è provata quella leva
  (85,9%).
- Si può anche non stimare nulla e imparare direttamente dei **voti**, alzando
  quello della leva appena tirata se ha reso più del solito e abbassandolo se ha
  reso meno. E il confronto "rispetto al solito" regge tutto: senza,
  aggiungendo quattro punti a tutte le ricompense il metodo crolla dall'84,1%
  al 48,5%.
- C'è poi una strada che di ogni leva non tiene un numero ma il **ventaglio** di
  quello che è ancora possibile crederne. Prima di ogni tiro si sorteggia un
  valore dentro ciascun ventaglio e si tira la leva che ha sorteggiato più alto:
  è il *Thompson sampling* del 1933. Il ventaglio si stringe da sé, e con lui
  l'esplorazione, senza niente da scegliere a mano (91,8%, il meglio della
  sezione). In cambio va dichiarato in anticipo quanto è largo il ventaglio di
  partenza, che vale come un certo numero di tiri finti: partire vaghi non costa
  niente (91,6%), partire convinti sì (75,2%), perché quei tiri finti ci mettono
  un'eternità a lasciarsi smentire.
- Non sono giocattoli: si incontrano nei test A/B che spostano i visitatori
  verso la versione che sta vincendo, nel decidere quali oggetti nuovi mostrare
  in un catalogo, e nel regolare le impostazioni di un modello prima di
  addestrarlo (ogni combinazione è una leva, provarla un po’ è un tiro). Il
  gradino successivo è il caso in cui prima di scegliere si guarda la
  situazione, ma le proprie scelte non la cambiano.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Un **bandit a più braccia** è un problema di decisione senza stato: $k$
  azioni, ricompense rumorose, e il solo dilemma fra esplorare e sfruttare. È
  un MDP con un solo stato.
- Le stime si aggiornano con *stima $\leftarrow$ stima $+$ passo $\cdot$
  errore*: con passo $1/n$ si ottiene la media, con passo costante una media
  pesata sul recente, che è ciò che serve se il problema **non è stazionario**.
  Le condizioni di **Robbins-Monro** ($\sum \alpha_n = \infty$,
  $\sum \alpha_n^2 < \infty$) separano i due casi, e torneranno per la
  convergenza del Q-learning.
- L'agente **avido** si chiude in una convinzione mai verificata (36,7% di
  scelte ottime sul banco di prova standard); $\varepsilon$-greedy lo risolve
  quasi a costo zero (80,2%) ma esplora **a casaccio**, e il suo $\varepsilon$
  va scelto guardando l'orizzonte: con $0{,}01$ il sorpasso su $0{,}1$ arriva
  passato il novemillesimo tiro.
- **Valori iniziali ottimisti** (86,6%, con passo fisso; 71,3% con la media
  campionaria, contro il 36,7% dell'avido puro): esplorazione quasi gratis,
  ma si esaurisce e non serve sui problemi non stazionari. **UCB** (85,9%):
  esplora di più le leve di cui sa di meno, e quello che perde per farlo
  cresce con il ritmo più lento possibile per una strategia che debba
  funzionare su qualunque insieme di leve (Lai e Robbins).
- Il **bandit a gradiente** impara preferenze invece di valori, ed è REINFORCE
  con baseline in miniatura. E la baseline regge tutto: traslando le
  ricompense di $+4$, senza di essa si passa dall'84,1% al 48,5%.
- Il **Thompson sampling** tiene una posteriore su ogni $q_*(a)$, ne estrae un
  campione per braccio e gioca l’$\arg\max$: la frequenza di gioco insegue
  così la probabilità a posteriori di essere il braccio ottimo
  (*probability matching*), e l'esplorazione si spegne da sola perché la
  varianza a posteriori va a zero, senza nessun $\varepsilon$ né $c$ da
  tarare. Coniugate: gaussiana sul banco di prova, Beta per ricompense in
  $\{0,1\}$. Sul banco di prova 91,8%; e per Bernoulli il suo rimpianto
  raggiunge **asintoticamente la costante** del limite di Lai e Robbins, che
  UCB1 avvicina soltanto (il banco di prova è però gaussiano: là il 91,8% è una
  misura). Il prezzo è che la garanzia è sul modello: un priore troppo stretto
  smette di esplorare mentre dovrebbe (75,2%), una $\sigma$ creduta troppo
  grande esplora invece per sempre, e la posteriore accumulata presuppone un
  problema **stazionario**.
- Si incontrano davvero in test A/B adattivi, esplorazione nei sistemi di
  raccomandazione e ricerca di iperparametri (Hyperband è *best-arm
  identification*). Il gradino successivo è il **bandit contestuale**, dove
  c'è uno stato ma le azioni non lo cambiano.
```

`````
