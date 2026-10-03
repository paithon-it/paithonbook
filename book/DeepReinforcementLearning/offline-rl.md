# Offline reinforcement learning: imparare da dati fissi

Nel 2018 un gruppo dell'Imperial College di Londra ha addestrato un sistema a
curare la sepsi, cioè un'infezione che dilaga in tutto l'organismo e fa crollare
la pressione. Lo hanno chiamato *AI Clinician*
{cite}`komorowski2018clinician`, e il suo compito era suggerire, per un
paziente in terapia intensiva, quanti liquidi somministrargli e quanto farmaco
dargli per tenere su la pressione.

Quel sistema non ha mai provato una terapia su un malato vero: ha imparato
guardando l'archivio dei ricoveri passati, con le decisioni prese dai medici e
ciò che ne è seguito. È l'unica strada possibile, del resto. Nessun comitato
etico autorizzerebbe un agente a somministrare farmaci «a caso» per esplorare,
come fa il Q-learning in un videogioco; e nessuno lo lascerebbe sterzare a caso
al volante di un'auto, o proporre acquisti assurdi a milioni di utenti per
misurarne la reazione.

Eppure è proprio così (provando, sbagliando, esplorando) che gli agenti dei
capitoli precedenti imparano. Il **reinforcement learning offline** (o *batch
RL*) affronta il caso opposto e più scomodo: imparare la migliore strategia
possibile *senza mai interagire con l'ambiente*, disponendo solo di un
registro di esperienze già accadute {cite}`levine2020offline`.

## Imparare da un archivio, non dall'esperienza

`````{tab} Elementare

Ai fornelli si impara assaggiando: aggiusti il sale, rifai il piatto, e ogni
tentativo ti dà un riscontro nuovo. Sui quaderni di ricette di tua nonna no: hai
solo un archivio chiuso (quello che lei ha cucinato negli anni, con gli esiti) e
da lì devi tirar fuori la ricetta migliore, senza poter provare nulla di
persona.

L'RL offline è la seconda situazione. Qualcuno, in passato, ha agito
nell'ambiente seguendo una sua strategia (un medico, un guidatore, un vecchio
algoritmo di raccomandazione) e ha lasciato un diario: «in questa situazione
ho fatto questo, ed è successo quello». L'agente riceve solo il diario. Non
può tornare indietro a chiedere «e se avessi fatto diversamente?». Deve
cavarsela con ciò che è scritto.

`````

`````{tab} Superiore

Formalmente disponiamo di un dataset fisso di transizioni

$$
\mathcal{D} = \big\{\, (s_i,\, a_i,\, r_i,\, s'_i) \,\big\}_{i=1}^{N},
$$

raccolto da una o più **policy comportamentali** $\pi_\beta$ (di cui in genere
non conosciamo la forma esplicita) e mai più ampliato: non c'è alcuna nuova
interazione con l'ambiente durante l'addestramento. L'obiettivo resta quello
consueto (trovare una policy $\pi$ che massimizzi il ritorno atteso scontato
$\mathbb{E}\big[\sum_t \gamma^t r_t\big]$) ma va raggiunto *dentro il
supporto* di $\mathcal{D}$. Il setting online concede qualcosa che qui manca:
il Q-learning e i metodi a gradiente di policy visti fin qui presuppongono di
poter *provare* le proprie ipotesi, e sono le prove a correggere le stime
sbagliate. Togliere le prove cambia la natura del problema, e cambia anche il
collaudo: una policy appresa offline va valutata sull'archivio (*off-policy
evaluation*). La stima più semplice ripesa i ritorni osservati con il rapporto
di importance sampling della {doc}`sezione sui metodi Monte Carlo
</ReinforcementLearning/monte-carlo>`, $\prod_t \pi(a_t\mid
s_t)/\pi_\beta(a_t\mid
s_t)$, che chiede di conoscere $\pi_\beta$ e ha una varianza che esplode con
l'orizzonte; l'alternativa, stimare $Q^\pi$ sui dati (*fitted Q evaluation*),
eredita l'errore di estrapolazione descritto più avanti. La valutazione
dell’*AI Clinician* è stata fatta così, ed è su questo punto che è stata
criticata: l'importance sampling non dà quasi peso alle traiettorie in cui la
policy appresa fa qualcosa di molto diverso dai medici, cioè proprio ai casi
difficili {cite}`jeter2019ai`.

`````

Verrebbe da pensare che sia facile, e la ragione l'abbiamo già vista nella
{doc}`sezione su DQN <dqn>`: il Q-learning è *off-policy*, cioè il suo bersaglio
valuta la policy greedy mentre i dati possono venire da un'altra policy. Se sa
fare questo, dovrebbe saper imparare anche da partite giocate da altri e finite
da un pezzo: basterebbe addestrarlo sull'archivio $\mathcal{D}$ invece che sulle
esperienze fresche. Fatto così, però, può fallire gravemente. Fujimoto e
colleghi lo mostrano su un compito di controllo continuo, perfino quando
l'archivio è quello che un altro agente sta raccogliendo e da cui lui stesso
impara con successo {cite}`fujimoto2019off`; con archivi enormi e vari, come i
cinquanta milioni di transizioni accumulati da un DQN in tutto il suo
addestramento su un gioco Atari, gli stessi metodi possono invece superare,
sulla maggior parte dei giochi, l'agente che li ha prodotti
{cite}`agarwal2020optimistic`. Capire *quando* e *perché* l'archivio rompe il
metodo è il cuore di tutto l'argomento.

## Il buco nero delle azioni mai viste

Il colpevole è la differenza fra le mosse che nell'archivio ci sono e le mosse
che l'agente, imparando, vorrebbe fare, la stessa frattura vista nell'imitazione
quando l'allievo usciva dalla fascia dimostrata. Ha un nome tecnico,
**distributional shift**, lo spostamento di distribuzione, e un modo molto più
chiaro di raccontarlo.

`````{tab} Elementare

Torniamo ai quaderni della nonna. Immagina che nei suoi decenni ai fornelli
non abbia *mai* messo il peperoncino in un piatto. Nel diario, alla voce
«peperoncino», non c'è nulla. Ora, se ti chiedessero «quanto sarebbe buono
questo risotto con tre cucchiai di peperoncino?», la risposta onesta è «non ne
ho idea». Ma un modello ingenuo, quando deve indovinare, tira comunque un
numero, e a volte, per puro caso della sua matematica interna, tira un numero
*altissimo*. «Risotto al peperoncino: buonissimo, dieci e lode.»

Ecco la trappola. L'agente cerca sempre l'azione col voto più alto. Se un voto
gonfiato per sbaglio spunta proprio su un'azione mai provata, l'agente ci si
butta a capofitto. E siccome non può cucinarla davvero per scoprire che è
immangiabile, quel voto sballato non viene mai corretto: anzi, contagia le stime
vicine e peggiora tutto. Nel gioco vero, in cui provi le mosse, l'errore si
smaschera da sé; qui no, e resta lì a fare danni.

`````

`````{tab} Superiore

Il target del Q-learning è $y = r + \gamma \max_{a'} Q_\theta(s', a')$. Il
problema è l'operatore $\max_{a'}$: spazia su *tutte* le azioni, comprese quelle
che $\pi_\beta$ non ha mai eseguito in $s'$. Su quelle azioni
*out-of-distribution* (OOD) la rete $Q_\theta$ non ha mai visto dati e non fa
che estrapolare; i suoi errori di estrapolazione sono casuali, ma il $\max$ non
è casuale: seleziona sistematicamente i valori più alti, cioè proprio le
sovrastime. Il target risulta gonfiato, la policy insegue quelle azioni
fantasma, e per bootstrapping il valore inflazionato si propaga all'indietro
agli altri stati.

Nel RL online questo circolo si spezza da solo: eseguendo l'azione
sopravvalutata si osserva la ricompensa vera, bassa, e la stima si corregge.
Offline il correttivo non arriva mai (non c'è nuova interazione) e l'errore può
crescere senza freni. Fujimoto e colleghi {cite}`fujimoto2019off` hanno mostrato
che questa **extrapolation error** è la ragione per cui gli algoritmi off-policy
standard possono crollare su dati fissi: DDPG addestrato su un archivio di un
milione di transizioni resta molto sotto l'agente che lo ha raccolto, anche
quando i due imparano dagli stessi dati, e le sue stime oscillano o divergono.
Non è però un destino: Agarwal e colleghi {cite}`agarwal2020optimistic` mostrano
che su archivi grandi e vari (tutta l'esperienza di un DQN, cinquanta milioni di
transizioni per gioco) QR-DQN e REM, due varianti di DQN che stimano
rispettivamente una distribuzione di ritorni e un insieme di teste di valore,
addestrati offline superano il DQN completamente addestrato sulla maggior parte
dei giochi Atari. Quanto conta l'estrapolazione dipende da quanto l'archivio
copre le azioni che la policy appresa vorrà fare. Tutte le tecniche che seguono
attaccano lo stesso nemico da angolazioni diverse: impedire, in un modo o
nell'altro, di fidarsi delle azioni fuori dal supporto dei dati.

`````

Prima di vedere i rimedi, tocchiamo con mano il fenomeno, e in miniatura. La
mossa è un numero solo, che va da $-1$ a $+1$: immagina lo sterzo. Il voto vero
di ogni posizione dello sterzo lo conosciamo noi, ed è costruito come una multa:
misura quanto ci si allontana dalla posizione perfetta, col segno cambiato. La
posizione perfetta è $+0{,}3$, e lì la multa è nulla e il voto vale $0$; ogni
altra posizione paga qualcosa, tanto più quanto più è lontana. Zero è il
massimo, insomma, e tutti gli altri voti sono numeri negativi.

Chi ha raccolto i dati, però, è stato prudente: ha provato soltanto la fetta fra
$-1{,}0$ e $-0{,}2$, quaranta volte in tutto. La mossa migliore, nell'archivio,
non c'è, e tutto l'esempio serve a mostrare che cosa succede per questo. Il
programma sullo sterzo fa passare una curva per quei quaranta punti e poi le
chiede
il voto di tutte le mosse, comprese quelle mai provate.

```python
import numpy as np
rng = np.random.default_rng(0)

# Valore-azione "vero": una parabola con l'ottimo in a = 0.3
def Q_vero(a):
    return -(a - 0.3) ** 2

# La policy comportamentale ha esplorato solo azioni "prudenti" in [-1.0, -0.2]:
# il dataset NON contiene mai l'azione ottima.
a_dati = rng.uniform(-1.0, -0.2, size=40)
q_dati = Q_vero(a_dati) + rng.normal(0, 0.01, size=a_dati.shape)

# Stimiamo Q con un modello flessibile (polinomio di grado 5) sui soli dati.
coeff = np.polyfit(a_dati, q_dati, deg=5)
Q_stima = np.poly1d(coeff)

azioni = np.linspace(-1, 1, 201)                 # candidate su TUTTO lo spazio

# (1) max naive: nessun vincolo, come nel Q-learning off-policy classico
a_naive = azioni[np.argmax(Q_stima(azioni))]
print(f"naive     : a*={a_naive:+.2f}  Q_stimato={Q_stima(a_naive):+.2f}  "
      f"Q_vero={Q_vero(a_naive):+.2f}")

# (2) max vincolato al supporto dei dati (idea alla base di BCQ)
in_supp = (azioni >= a_dati.min()) & (azioni <= a_dati.max())
a_vinc = azioni[in_supp][np.argmax(Q_stima(azioni[in_supp]))]
print(f"vincolato : a*={a_vinc:+.2f}  Q_stimato={Q_stima(a_vinc):+.2f}  "
      f"Q_vero={Q_vero(a_vinc):+.2f}")

# (3) e con una retta al posto della curva flessibile, senza vincolo
Q_retta = np.poly1d(np.polyfit(a_dati, q_dati, deg=1))
a_retta = azioni[np.argmax(Q_retta(azioni))]
print(f"retta     : a*={a_retta:+.2f}  Q_stimato={Q_retta(a_retta):+.2f}  "
      f"Q_vero={Q_vero(a_retta):+.2f}")
```

```text
naive     : a*=+1.00  Q_stimato=+30.11  Q_vero=-0.49
vincolato : a*=-0.21  Q_stimato=-0.25  Q_vero=-0.26
retta     : a*=+1.00  Q_stimato=+2.06  Q_vero=-0.49
```

Il voto più alto cade sullo sterzo tutto a destra, dove nessuno ha mai messo
piede: lì la curva promette $+30{,}1$, e il valore vero è $-0{,}49$. Trenta,
quando il voto migliore che esista vale zero.

```{figure} ../figures/supporto-dei-dati.svg
:name: fig-supporto-dei-dati
:alt: "Grafico del voto in funzione della posizione dello sterzo, da meno uno a più uno. Una fascia ocra copre il tratto da −1,00 a −0,20, il supporto dei dati, e dentro ci sono i quaranta punti osservati. La curva teal del voto vero è una parabola che ha il suo massimo, zero, in +0,30, cerchiato perché quella mossa nell’archivio non c’è. La curva terracotta stimata sui quaranta punti li segue dentro la fascia, e appena ne esce impenna e lascia il riquadro, salendo fino a +30,1 in più uno, dove il voto vero è −0,49. Cercando il massimo dentro la fascia si sceglie invece −0,21, dove stima −0,25 e valore vero −0,26 coincidono."
:width: 92%

Dentro la fascia ocra, dove i quaranta punti stanno, la curva stimata e quella
vera si sovrappongono; appena fuori la stimata impenna e lascia il riquadro.
Cercare il voto più alto su tutto l'asse porta allo sterzo tutto a destra, con
una promessa di $+30{,}1$ a fronte di un valore vero di $-0{,}49$; cercarlo
dentro la fascia porta a $-0{,}21$, dove promessa e valore coincidono.
```

Non c'è niente di misterioso, e {numref}`fig-supporto-dei-dati` fa vedere dove
succede. Una curva passa docilmente in mezzo ai quaranta punti che ha, e fuori
da quel tratto prosegue come le pare: non ha più nessun dato che la trattenga,
e le basta una piccola pendenza sbagliata al bordo per schizzare in alto nel
giro di un intervallo. La flessibilità decide quanto forte è lo schizzo, non se
ci sarà: rifacendo lo stesso conto con una retta al posto della curva, il voto
più alto cade di nuovo sullo sterzo tutto a destra, con una promessa di
$+2{,}06$ dove il migliore possibile vale zero. Al riparo si sarebbe soltanto
scegliendo esattamente la forma giusta, che qui è la parabola; ma la forma
giusta è quella che non si conosce, perché è ciò che si sta cercando.

Vincolando invece la ricerca alla zona che i dati coprono davvero (il loro
**supporto**, che con quaranta punti è il tratto che li racchiude) si sceglie
$-0{,}21$, e stima ($-0{,}25$) e realtà ($-0{,}26$) tornano a coincidere. È, in
miniatura, la prima famiglia di soluzioni, e mostra anche che cosa costa: la
mossa perfetta, $+0{,}3$, resta irraggiungibile, perché nell'archivio non c'è.
Si è rinunciato al meglio in cambio di non prendere lucciole per lanterne, ed è
un baratto che l'RL offline fa continuamente.

## BCQ: restare vicini a ciò che è stato visto

L'idea è quella suggerita dall'esempio, e più semplice di così non si può: se il
guaio nasce dal dare un voto a mosse mai provate, allora non diamoglielo. Si
cercherà il voto più alto soltanto fra le mosse che nell'archivio compaiono
davvero.

Il primo algoritmo a farlo nel controllo continuo, pensato apposta per il RL
offline con le reti
profonde, è **BCQ** (*Batch-Constrained deep Q-learning*), di Scott Fujimoto,
David Meger e Doina Precup {cite}`fujimoto2019off`, presentato nel 2019.

`````{tab} Elementare

Come fa BCQ a sapere quali mosse siano plausibili? Impara a imitare chi ha
raccolto i dati. Da una parte addestra una rete a rispondere alla domanda «in
una serata come questa, che cosa avrebbe cucinato la nonna?», e quella rete
propone una manciata di ricette possibili, tutte del genere che nei quaderni
compare davvero. Dall'altra parte la rete dei voti giudica soltanto quelle,
e si tiene la migliore.

La differenza è tutta qui: alle ricette che nessuno ha mai scritto non viene mai
chiesto un voto, quindi nessun voto di fantasia può vincere, perché non è mai
stato dato.

E allora BCQ non può fare altro che ripetere la nonna? No, e per due motivi. Il
primo è che fra le ricette plausibili può scegliere la *migliore*, mentre la
nonna a volte sbagliava; già questo basta a fare meglio di lei. Il secondo è che
a ciascuna ricetta proposta è concesso un ritocco piccolo: quanto ritoccare lo
impara una rete, ma fin dove le sia concesso spingersi è un numero che si
sceglie a mano, prima di cominciare, e alzarlo o abbassarlo sposta BCQ fra il
copiare la nonna e il fidarsi dei voti.
Il ritocco è un rischio, certo, ma di taglia controllata: si resta
accanto a qualcosa che è stato davvero cucinato, e non si inventa un piatto in
mezzo al nulla.

`````

`````{tab} Superiore

Per sapere quali azioni siano plausibili, BCQ addestra un modello
generativo (un *variational autoencoder*: una rete che strozza i dati in
poche variabili e da quelle li ricostruisce, derivata per esteso nel capitolo
sui modelli latenti) sulle coppie $(s, a)$ del
dataset: dato uno stato, genera azioni simili a quelle che $\pi_\beta$ avrebbe
scelto in situazioni analoghe. La rete $Q$ viene poi massimizzata solo su un
pugno di azioni campionate da questo generatore, ciascuna con una piccola
perturbazione appresa che concede un margine di miglioramento:

$$
\pi(s) = \operatorname*{arg\,max}_{a_i + \xi_\phi(s, a_i, \Phi)}
Q_\theta\big(s,\, a_i + \xi_\phi(s, a_i, \Phi)\big),
\qquad a_i \sim G_\omega(s),\quad i = 1, \dots, n,
$$

dove $G_\omega$ è il generatore, $\xi_\phi$ la perturbazione, che non può
superare in ampiezza $\Phi$, e $n$ il numero di azioni candidate (nel lavoro
$n = 10$). Le due manopole spostano BCQ fra due estremi: con $\Phi = 0$ e
$n = 1$ è una clonazione comportamentale, con $\Phi$ pari a tutta l'ampiezza
delle azioni e $n \to \infty$ torna il Q-learning su tutto lo spazio.
L'operatore di massimizzazione non può più cadere nelle regioni fantasma:
sceglie il meglio *tra ciò che si sarebbe davvero potuto fare*.

`````

È il modo più letterale di rispondere al distributional shift: costruire un
recinto attorno ai dati e non uscirne.

## CQL: essere pessimisti sull'ignoto

BCQ mette un recinto attorno alle mosse ammesse, e per farlo gli serve una
seconda rete, quella che imita chi ha raccolto i dati. Si può ottenere lo stesso
effetto senza quella seconda rete, e nel 2020 Aviral Kumar, Aurick Zhou, George
Tucker e Sergey Levine mostrano come: **CQL** (*Conservative Q-Learning*)
{cite}`kumar2020conservative`. Invece di vietare le azioni mai viste (in gergo
*out-of-distribution*, «fuori distribuzione», abbreviato **OOD**), le rende
*poco appetibili*, intervenendo direttamente sul numero che l'addestramento
cerca di far scendere (la *loss*, la misura di quanto la rete sta sbagliando).

`````{tab} Elementare

Un critico gastronomico prudente si dà una regola: dei piatti documentati nei
quaderni si fida di ciò che c'è scritto, di quelli che non ha mai assaggiato dà
per scontato il peggio. Il pessimismo però non lo distribuisce a caso. Va a
cercare i piatti mai provati che sulla carta gli sembrano più promettenti: sono
quelli che rischiano di trascinarlo in una serata storta, e sono quelli che
ritocca all'ingiù.

CQL insegna questa prudenza alla rete dei voti. Una rete non impara in un colpo
solo: si corregge migliaia di volte, un pochino per volta, e a ogni correzione
CQL aggiunge due spinte. Abbassa i voti delle mosse che nel diario non
compaiono, e alza quelli delle mosse che ci sono davvero. Ne esce un sistema di
voti *conservativo*, pessimista su tutto ciò che non ha visto.

Quanto forte spinge è una manopola, e la si gira a mano. Girata poco, la
prudenza resta un'inclinazione: un piatto mai assaggiato che il critico si
figura strepitoso può ancora finire in cima alla lista. Girata a fondo, tutto
ciò che non è documentato sprofonda, e allora il critico dai quaderni non esce
più, nemmeno quando fuori ci sarebbe di meglio.

La promessa che si riesce a dimostrare, poi, è più modesta della regola. Prendi
una serata e i piatti che il critico ordinerebbe: la media dei voti che dà a
quei piatti non supera mai quello che la serata renderà davvero, se si ordina
come dice lui. Un singolo piatto dentro il gruppo può ancora essere
sopravvalutato, e può ancora battere un piatto documentato. La garanzia è sulla
media, non su ogni riga della lista, e vale su una serata come quelle che i
quaderni raccontano: su una serata che nei quaderni non c'è non è promesso
niente.

E vale a due condizioni. La prima è che la manopola sia girata abbastanza, e
quanto basti dipende da quanto ciascun piatto è documentato: uno cucinato una
volta sola dà un voto malfermo e chiede una spinta all'ingiù più robusta di uno
cucinato cento volte. La seconda riguarda il modo in cui il critico mette
insieme i piatti che si somigliano. Finché ogni piatto ha la sua scheda, o le
somiglianze il critico le ha decise una volta per tutte («i risotti con i
risotti, le zuppe con le zuppe»), abbassare il voto di un piatto immaginato non
tocca gli altri, e la promessa regge. Un critico vero, però, che cosa somiglia
a che cosa lo impara strada facendo, e allora la spinta che abbassa un piatto
immaginato scivola anche su quelli che gli somigliano, compresi quelli scritti
nei quaderni: la prudenza continua a funzionare in pratica, la dimostrazione lì
non la segue più.

`````

`````{tab} Superiore

CQL aggiunge alla consueta minimizzazione dell'errore di Bellman un termine
regolarizzatore che comprime i valori delle azioni fuori distribuzione:

$$
\min_{Q}\ \alpha\Big(
\underbrace{\mathbb{E}_{s\sim\mathcal{D},\, a\sim\mu(\cdot\mid
s)}\big[Q(s,a)\big]}_{\text{abbassa: azioni OOD}}
-
\underbrace{\mathbb{E}_{(s,a)\sim\mathcal{D}}\big[Q(s,a)\big]}_{\text{alza:
azioni nel dataset}}
\Big)
+ \tfrac{1}{2}\,\mathbb{E}_{\mathcal{D}}\Big[\big(Q - \hat{\mathcal{B}}Q\big)^2\Big].
$$

Qui $\mu(\cdot\mid s)$ è una distribuzione *ampia* di azioni candidate (nella
pratica una softmax sui $Q$ stessi, o l'uniforme), $\hat{\mathcal{B}}Q$ è il
target di Bellman calcolato sull'iterata precedente e $\alpha>0$ dosa la
conservatività. Il primo termine tira
*giù* i $Q$ delle azioni pescate da $\mu$ (tipicamente OOD), mentre il secondo
li tira *su* sulle azioni realmente presenti in $\mathcal{D}$. Kumar e
colleghi dimostrano che, per $\alpha$ abbastanza grande e con $\mu$ agganciata
alla policy che si sta valutando, il valore atteso delle azioni sotto la
$Q$ così ottenuta minora il vero valore della policy in ogni stato del
dataset: un limite
inferiore *in valore*, non punto per punto (la singola stima $Q(s,a)$ può
ancora eccedere quella vera), e su uno stato mai visto non promesso affatto.

La prudenza sull'ignoto resta quindi una garanzia formale e non un'euristica,
ma conviene dire dove la garanzia vive. L'enunciato principale è nel caso
tabellare, e vale per un
$\alpha$ abbastanza grande rispetto all'errore di campionamento, che dipende da
quante volte la coppia $(s,a)$ compare nel dataset. In appendice gli autori lo
estendono agli approssimatori lineari e alle reti nel regime del *neural tangent
kernel*, cioè con le caratteristiche di fatto congelate: due ipotesi che una
rete addestrata normalmente non soddisfa, e infatti l'analisi della CQL con le
reti profonde vere gli autori la lasciano scritta come lavoro da fare. Quello
che resta è una buona euristica con un teorema alle spalle, e la portata del
teorema è quella appena detta.

`````

## IQL: non guardare mai fuori dai dati

CQL valuta ancora le azioni OOD, salvo poi penalizzarle. Nel 2022 Ilya
Kostrikov, Ashvin Nair e Sergey Levine portano l'idea alle estreme conseguenze
con **IQL** (*Implicit Q-Learning*) {cite}`kostrikov2022offline`: costruire
una strategia migliore di quella che ha raccolto i dati *senza mai chiedere alla
rete dei voti che voto darebbe a una mossa che nell'archivio non c'è*. Se non
guardi mai fuori, non puoi essere ingannato da ciò che c'è fuori.

«Implicito» è il nome di quello che non si fa. Fin qui, per sapere quanto vale
una situazione, si prendeva il voto della mossa migliore fra tutte quelle
possibili: è lì che entravano le mosse mai provate. IQL quel confronto non lo
fa mai; il numero che ne sarebbe uscito lo ottiene per un'altra strada, come
sottoprodotto di una stima fatta sui soli dati, e la mossa fantasma non viene
nemmeno nominata.

`````{tab} Elementare

Come si fa a scegliere bene senza mai considerare piatti mai cucinati? IQL
cambia la domanda che rivolge ai quaderni. Non chiede più «quanto varrebbe
questa ricetta ipotetica?», che è la domanda da cui nascono i voti di fantasia:
chiede «nelle serate come questa, fin dove sono arrivate le ricette davvero
provate, contando di più quelle andate meglio?», e quella risposta è il metro
della situazione. È come giudicare il potenziale di una cucina dai suoi piatti
più riusciti, senza fantasticare su menù mai esistiti.

I due numeri si calcolano uno dall'altro. Il voto di una ricetta tiene conto del
metro della situazione in cui ti lascia; il metro di una situazione esce dai
voti delle ricette provate lì. Aggiornarli insieme, nello stesso istante, dà due
conti che si rincorrono e non si posano più. Il rimedio è tenere sul tavolo una
copia dei voti stampata poco prima, come la copia congelata di DQN, e
ricalcolare il metro leggendo quella: serve qualcosa di fermo a cui appoggiarsi,
e chi salta l'accorgimento si ritrova con un metodo che non impara niente.

Con quel metro («quanto si ottiene qui scegliendo bene fra ciò che è stato
fatto») si rileggono poi le
pagine del diario: le mosse che hanno reso più del solito nella loro
situazione vengono imitate di più, le altre di meno. Dall'inizio alla fine,
nessuna azione fuori dal diario viene mai nemmeno nominata: dove gli altri
metodi mettono recinti o penalità, IQL toglie proprio l'occasione di
sbagliare.

`````

`````{tab} Superiore

Il trucco sta nel sostituire il $\max_{a'} Q(s',a')$ (che costringe a spaziare
su tutte le azioni) con una funzione valore $V(s')$ stimata in modo obliquo,
tramite **regressione expectile**. Un expectile alto (vicino a $1$) fa sì che
$V(s)$ approssimi il *massimo* dei valori $Q(s,a)$ ma solo sulle azioni $a$
*effettivamente osservate* in quello stato: coglie «il meglio tra ciò che è
stato fatto» senza mai nominare un'azione ipotetica. Con i valori di $\tau$ che
si usano in pratica la stima sta fra la media e il massimo, tanto più vicina al
massimo quanto più $\tau$ si avvicina a $1$. Le tre reti si addestrano
per pura regressione sui dati:

$$
\mathcal{L}_V = \mathbb{E}_{(s,a)\sim\mathcal{D}}\Big[ L_2^{\tau}\big(Q(s,a) -
V(s)\big)\Big],
\qquad
L_2^{\tau}(u) = \big|\,\tau - \mathbb{1}(u<0)\,\big|\; u^2,
$$

$$
\mathcal{L}_Q = \mathbb{E}_{(s,a,s')\sim\mathcal{D}}\Big[\big(r + \gamma\,
V(s') - Q(s,a)\big)^2\Big].
$$

Dove $\tau \in (0,1)$ è l’expectile (in pratica $\tau \approx 0{,}7$–$0{,}9$;
questo $\tau$ è il livello dell'asimmetria, e non ha niente a che vedere né con
la traiettoria né con il peso dello scorrimento di DDPG):
la perdita asimmetrica $L_2^\tau$ pesa di più i residui positivi, spingendo $V$
verso l'alto della distribuzione dei $Q$ nel dataset. Un dettaglio che sembra di
implementazione e non lo è: il $Q$ dentro $\mathcal{L}_V$ è una copia
ritardata, come la rete-target di DQN. Senza, le due regressioni si
inseguirebbero a vicenda senza niente di fermo a cui aggrapparsi, ed è il punto
esatto in cui una reimplementazione di IQL smette di funzionare. Il target di
$\mathcal{L}_Q$
usa $V(s')$, non un massimo su azioni arbitrarie: ecco perché nessuna azione
OOD viene mai valutata. La policy si estrae infine per *advantage-weighted
regression*, imitando le azioni del dataset con peso
$\exp\!\big(\beta_{\text{AWR}}\,(Q(s,a)-V(s))\big)$: l'esponenziale tiene i
pesi positivi anche sui vantaggi negativi, e $\beta_{\text{AWR}}$, l'inverso di
una temperatura (che con la policy comportamentale $\pi_\beta$ ha in comune
solo la lettera), decide quanto la policy assomigli a quella che ha raccolto i
dati ($\beta_{\text{AWR}} \to 0$) o a quella che insegue il vantaggio massimo
($\beta_{\text{AWR}}$ grande). Nell'implementazione degli autori i pesi sono
troncati a $100$, perché l'esponenziale non faccia esplodere un singolo
campione.

`````

Il nucleo di IQL, in PyTorch, sta in poche righe. È una misura d'errore
volutamente sbilanciata: con `tau=0.8` una stima troppo bassa viene rimproverata
quattro volte più di una stima troppo alta ($0{,}8$ contro $0{,}2$). E siccome
sbagliare per difetto costa quattro volte di più, alla stima conviene stare
sopra la media dei voti osservati, ma non sul migliore: con voti sparsi in modo
uniforme fra $0$ e $1$ si posa a $0{,}667$, e al massimo si avvicina soltanto
quando $\tau$ si avvicina a $1$.

```python
import torch

def expectile_loss(q, v, tau=0.8):
    # regressione expectile: residui positivi pesati di piu (tau > 0.5).
    # detach(): q e' la copia ferma, l'aggiornamento tocca solo v.
    diff = q.detach() - v
    peso = torch.where(diff > 0, tau, 1.0 - tau)
    return (peso * diff.pow(2)).mean()

# dove si posa la stima: si provano i candidati e si tiene il piu' economico
voti = torch.linspace(0, 1, 10_001, dtype=torch.float64)   # uniformi fra 0 e 1
candidati = torch.linspace(0, 1, 1_001, dtype=torch.float64)
for tau in (0.5, 0.8, 0.9, 0.99):
    perdite = torch.stack([expectile_loss(voti, v, tau) for v in candidati])
    migliore = candidati[perdite.argmin()]
    print(f"tau={tau:4.2f}: la stima si posa a {migliore:.3f}")
```

```text
tau=0.50: la stima si posa a 0.500
tau=0.80: la stima si posa a 0.667
tau=0.90: la stima si posa a 0.750
tau=0.99: la stima si posa a 0.909
```

## Decision Transformer: l'RL come previsione di sequenze

Fin qui abbiamo curato il Q-learning perché sopravvivesse ai dati fissi. Nel 2021
un gruppo di Berkeley (Lili Chen, Kevin Lu e colleghi) propone di cambiare
proprio domanda {cite}`chen2021decision`: e se smettessimo di stimare valori e
trattassimo l'apprendimento offline come un problema di modellazione di
sequenze, lo stesso su cui eccellono i Transformer, il tipo di rete con
cui oggi si costruiscono i modelli linguistici?

`````{tab} Elementare

Un modello linguistico impara a completare le frasi: gli dài l'inizio e lui
predice la parola dopo, avendo letto miliardi di testi. Il **Decision
Transformer** fa lo stesso, ma la «frase» è una partita: una sequenza di
situazioni e mosse. Lo alleni sui diari delle partite passate a completare la
sequenza: «dopo questa situazione, che mossa è stata fatta?».

C'è un ingrediente in più, ed è geniale nella sua semplicità. All'inizio di
ogni mossa gli si dice anche *quanto punteggio si vuole ancora totalizzare da
qui alla fine*. In addestramento questo numero è noto (basta sommare le
ricompense future). Al momento di giocare davvero, gli si mette in bocca un
obiettivo ambizioso («da qui voglio fare cento punti») e il modello, per
coerenza con tutte le partite viste, produce le mosse che *tipicamente portano
a cento punti*. A ogni mossa il traguardo si aggiorna da sé: incassati venti
punti, da lì in avanti la richiesta diventa ottanta. Non calcola alcun valore:
racconta la partita che vorresti, e la recita.

Le mosse le sa imitare, le situazioni no. Se una mossa storta lo porta in una
posizione che nei diari non compare, una mossa la produce lo stesso, ma sta
rispondendo a una domanda che non ha mai visto; e siccome la posizione dopo
dipende da quella risposta, lo scarto si allarga a ogni passo invece di
rientrare.

`````

`````{tab} Superiore

Il Decision Transformer riordina la traiettoria come una sequenza di token

$$
\big(\hat{R}_1,\, s_1,\, a_1,\ \hat{R}_2,\, s_2,\, a_2,\ \dots,\ \hat{R}_T,\,
s_T,\, a_T\big),
\qquad
\hat{R}_t = \sum_{t'=t}^{T} r_{t'},
$$

dove $\hat{R}_t$ è il **return-to-go**, la somma delle ricompense da $t$ in
poi. Un Transformer causale in stile GPT (la stessa architettura ad
auto-attenzione mascherata descritta nel {doc}`capitolo sui Transformer
</Transformers/overview>`) predice in modo autoregressivo l'azione $a_t$
condizionando sui token precedenti, cioè su return desiderato, stati e azioni
fino a $s_t$. L'addestramento è puramente supervisionato: minimizza
l'errore (cross-entropy per azioni discrete, MSE per continue) tra l'azione
predetta e quella nel dataset. È clonazione comportamentale, cioè imitazione
delle azioni osservate, con in più il condizionamento sul ritorno desiderato.
Non compaiono né equazione di Bellman, né bootstrapping, né operatore $\max$, e
quindi neppure la sovrastima delle azioni OOD che affligge il Q-learning
offline. In cambio eredita la fragilità della clonazione: la distribuzione
degli stati resta quella di chi ha prodotto il dataset, non quella indotta
dalla politica appresa.

In fase di controllo si fissa un return-to-go iniziale $\hat{R}_1$ desiderato, si
osserva lo stato, si genera l'azione; a ogni passo si decrementa il return-to-go
della ricompensa incassata e si prosegue. È il ponte esplicito tra reinforcement
learning e *sequence modeling*: condizionare sul risultato voluto, anziché
inseguire un valore stimato. Che a contare sia soprattutto il condizionamento,
più dell'architettura, lo suggerisce un confronto successivo: sui compiti di
locomozione simulata una semplice rete densa condizionata sul ritorno eguaglia
il Decision Transformer {cite}`emmons2022rvs`.

`````

Il Decision Transformer non è sempre il migliore, e il caso in cui perde si
capisce bene: se nell'archivio non c'è una sola partita andata a finire bene,
lui non ha niente da recitare, perché sa soltanto ripetere partite che ha visto.

CQL e IQL invece i voti li stimano, e questo dà loro un potere in più: sanno
«ricucire». Un voto sta attaccato a una singola mossa in una singola situazione,
non a una partita intera; quindi se due partite mediocri passano tutte e due per
la stessa situazione, l'inizio buono della prima si può attaccare al finale
buono della seconda, e ne esce un percorso migliore di entrambe, che
nell'archivio non c'è. Il Decision Transformer, che le partite le racconta
intere, di quella cucitura non ha un meccanismo. Gli autori la vedono succedere
lo stesso su un piccolo problema di percorsi in un grafo, dove circa un sesto
dei percorsi generati non compare in nessuna partita dell'archivio
{cite}`chen2021decision`; ma lì è un effetto osservato, mentre nei metodi a
valori discende dalla forma stessa del voto.

Il Decision Transformer ha comunque aperto, insieme al Trajectory Transformer di
Janner e colleghi, uscito il giorno dopo {cite}`janner2021offline`, una linea
feconda: una parte del reinforcement learning si può riformulare come
apprendimento supervisionato di sequenze, condizionato sul risultato che si
vuole.

## Un filo che torna: le preferenze dell'RLHF

Questa prospettiva illumina qualcosa che abbiamo già incontrato. In fondo alla
{doc}`sezione sulla ricerca ad albero <mcts-alphago>` abbiamo visto l’RLHF,
il modo in cui si addestrano oggi gli assistenti conversazionali
{cite}`ouyang2022training`: delle persone confrontano a due a due le risposte
del modello e dicono quale preferiscono; da quei confronti si costruisce un
giudice automatico, il *modello di ricompensa*, che da lì in poi assegna i voti
al posto loro.

Quei confronti formano un archivio, e come ogni archivio resta fermo mentre
l'addestramento va avanti. Fermo però non vuol dire chiuso
per sempre: Ouyang e colleghi scrivono che i due passi
(raccogliere confronti e ottimizzare) si possono iterare di continuo,
tornando dalle persone a chiedere nuovi giudizi sulle risposte della versione
migliore del momento. Si raccoglie a tornate. Ma fra una tornata e l'altra il
giudice non cambia di una virgola, e il grosso dei confronti su cui è stato
costruito viene da risposte di versioni precedenti.

Tanto basta perché l'RLHF erediti le tensioni di tutta la sezione. Il modello si
allontana dalle risposte che le persone hanno davvero giudicato, e proprio là
dove nessuno ha giudicato niente il giudice automatico è più facile da
imbrogliare: è il risotto al peperoncino, con altri ingredienti. E il rimedio è
della stessa famiglia: al punteggio che il modello cerca di far salire si
sottrae una penalità, che cresce man mano che le sue risposte si allontanano da
quelle di partenza. Allontanarsi resta possibile, ma costa, ed è una cugina
diretta della prudenza di BCQ e di CQL.

Anzi, il distinguo spiega proprio *perché* si itera. Ottimizzando, il modello si
sposta fuori dalla zona che i vecchi confronti coprivano, e quei confronti
valgono sempre meno; tornare dalle persone serve a farsi dire come sono le
risposte di adesso, non quelle di prima. Imparare da dati fissi, dalla terapia
intensiva agli assistenti conversazionali, pone sempre la stessa domanda: quanto
possiamo fidarci di ciò che non abbiamo mai visto?

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Il RL offline impara da un archivio chiuso di esperienze già accadute,
  come i quaderni di ricette della nonna: qualcun altro ha agito e ha lasciato
  scritto com'è andata. Nessun assaggio nuovo, nessuna prova sul campo.
- Il metodo classico, applicato così com'è, può fallire gravemente (con archivi
  enormi e vari a volte regge). L'agente cerca sempre il voto più alto, e i voti
  più alti finiscono per capitare proprio sulle mosse mai provate, quelle su cui
  il modello può solo tirare a indovinare (il risotto al peperoncino da dieci e
  lode). Nessuna prova può smentire quel voto gonfiato, che anzi contagia le
  stime vicine.
- BCQ costruisce un recinto: valuta solo le mosse plausibili secondo
  l'archivio, generate imitando chi i dati li ha raccolti. CQL non vieta
  nulla, insegna prudenza: abbassa i voti di ciò che non è mai stato provato e
  alza quelli di ciò che è documentato. La garanzia che se ne ricava è sulla
  media delle mosse che poi sceglierà, non su ogni singolo voto: uno gonfiato
  può ancora scapparci, e nelle situazioni che l'archivio non contiene non è
  promesso niente. IQL toglie proprio l'occasione di sbagliare: chiede
  solo quanto hanno reso, in situazioni come questa, le mosse migliori fra
  quelle davvero fatte, e non nomina mai un'azione fuori dal diario.
- Il Decision Transformer cambia domanda: tratta la partita come una frase
  da completare e si allena sui diari a predire la mossa successiva. Gli si
  dice anche quanto punteggio si vuole ancora totalizzare, e lui produce le
  mosse che di solito portano lì. Nessun voto da stimare, quindi nessun voto
  gonfiato.
- Anche i confronti con cui le persone giudicano le risposte di un assistente
  conversazionale sono un archivio, che si riapre a tornate ma resta fermo
  mentre l'addestramento va avanti: il giudice costruito su quei confronti si
  inganna più facilmente proprio dove nessuno ha giudicato niente, e il rimedio
  è della stessa famiglia, una penalità che tiene il modello vicino a com'era
  all'inizio.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Il RL offline (batch RL) impara da un dataset fisso di transizioni
  $\mathcal{D}=\{(s,a,r,s')\}$ raccolto da una policy comportamentale
  $\pi_\beta$, senza alcuna nuova interazione con l'ambiente.
- Il Q-learning off-policy naive fallisce per distributional shift:
  l'operatore $\max$ sovrastima le azioni fuori distribuzione (errore di
  estrapolazione), la policy le insegue e (senza prove che smascherino
  l'errore) la sovrastima si propaga {cite}`fujimoto2019off`.
- BCQ vincola le azioni valutate a quelle plausibili secondo un modello
  generativo del dataset; CQL aggiunge un termine conservativo che abbassa i
  $Q$ delle azioni OOD e alza quelli nel dataset, ottenendo un limite inferiore
  del valore negli stati del dataset, non punto per punto
  {cite}`kumar2020conservative`; IQL stima $V$ per regressione
  expectile e non interroga mai azioni fuori dai dati
  {cite}`kostrikov2022offline`.
- Il Decision Transformer riformula l'RL come modellazione di sequenze:
  condiziona sul *return-to-go* desiderato e predice l'azione con un Transformer,
  in modo puramente supervisionato {cite}`chen2021decision`.
- I dati di preferenza dell’RLHF sono anch'essi un dataset raccolto a
  tornate e fermo fra una tornata e l'altra: il modello di ricompensa viene
  interrogato fuori dal suo supporto e si lascia sfruttare
  {cite}`gao2023scaling`, e la penalità KL verso il modello di partenza è un
  rimedio della stessa famiglia {cite}`ouyang2022training`.
```
`````
