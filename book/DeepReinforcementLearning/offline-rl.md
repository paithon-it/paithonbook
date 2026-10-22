# Offline reinforcement learning: imparare da dati fissi

Nel 2018 un gruppo dell'Imperial College di Londra ha addestrato un sistema a
curare la sepsi, cioè un'infezione che dilaga in tutto l'organismo e fa crollare
la pressione. Lo hanno chiamato *AI Clinician*
{cite}`komorowski2018clinician`, con il compito di suggerire, per un
paziente in terapia intensiva, quanti liquidi e quanto farmaco
somministrare per sostenere la pressione.

Quel sistema non ha mai provato una terapia su un malato vero, ma ha imparato
guardando l'archivio dei ricoveri passati, con le decisioni prese dai medici e
ciò che ne è seguito. Del resto è l'unica strada possibile, perché nessun
comitato etico autorizzerebbe un agente a somministrare farmaci «a caso» per
esplorare, come fa il Q-learning in un videogioco. Allo stesso modo, nessuno lo
lascerebbe sterzare a caso al volante di un'auto, o proporre acquisti assurdi a
milioni di utenti per misurarne la reazione.

Eppure gli agenti del Q-learning e del gradiente di policy imparano proprio
così: provando, sbagliando, esplorando. Il **reinforcement learning offline**
(o *batch RL*) affronta il caso opposto e più scomodo: imparare la migliore
strategia possibile *senza mai interagire con l'ambiente*, disponendo solo di
un registro di esperienze già accadute {cite}`levine2020offline`.

## Imparare da un archivio, non dall'esperienza

`````{tab} Elementare

Ai fornelli si impara assaggiando: aggiusti il sale, rifai il piatto, e ogni
tentativo ti dà un riscontro nuovo. Sui quaderni di ricette di tua nonna no: hai
solo un archivio chiuso (quello che lei ha cucinato negli anni, con gli esiti) e
da lì devi tirar fuori la ricetta migliore, senza poter provare nulla di
persona.

L'RL offline è la seconda situazione. Qualcuno, in passato, ha agito
nell'ambiente seguendo una sua strategia (ad esempio un medico, un guidatore o
un vecchio algoritmo di raccomandazione) e ha lasciato un diario: «in questa
situazione ho fatto questo, ed è successo quello». L'agente riceve solo il
diario. Non può tornare indietro a chiedere «e se avessi fatto diversamente?»,
quindi deve cavarsela con ciò che è scritto.

`````

`````{tab} Superiore

Formalmente disponiamo di un dataset fisso di transizioni

$$
\mathcal{D} = \big\{\, (s_i,\, a_i,\, r_i,\, s'_i) \,\big\}_{i=1}^{N},
$$

raccolto da una o più **policy comportamentali** $\pi_\beta$ (di cui in genere
non conosciamo la forma esplicita) e mai più ampliato; in altri termini,
durante l'addestramento non c'è alcuna nuova interazione con l'ambiente.
L'obiettivo resta quello consueto (trovare una policy $\pi$ che massimizzi il
ritorno atteso scontato $\mathbb{E}\big[\sum_t \gamma^t r_t\big]$), ma va
raggiunto *dentro il supporto* di $\mathcal{D}$.

Il setting online concede qualcosa che qui manca. Il Q-learning e i metodi a
gradiente di policy visti fin qui presuppongono infatti di poter *provare* le
proprie ipotesi, e sono le prove a correggere le stime sbagliate. Togliere le
prove cambia la natura del problema, e cambia anche la valutazione: una
policy appresa offline va valutata sull'archivio, ed è il problema della
**off-policy evaluation**.

La stima più semplice ripesa i ritorni osservati con il rapporto
di importance sampling della {doc}`sezione sui metodi Monte Carlo
</ReinforcementLearning/monte-carlo>`, $\prod_t \pi(a_t\mid
s_t)/\pi_\beta(a_t\mid s_t)$, ma richiede di conoscere $\pi_\beta$ e ha una
varianza che esplode con l'orizzonte. L'alternativa, cioè stimare $Q^\pi$ sui
dati (*fitted Q evaluation*), eredita l'errore di estrapolazione descritto in
«Il buco nero delle azioni mai viste». La valutazione di *AI Clinician*
ricorreva all'importance sampling, e per questo è stata criticata: il metodo dà
un peso
quasi nullo alle traiettorie in cui la policy appresa fa qualcosa di molto
diverso dai medici, quale che ne fosse l'esito per il paziente
{cite}`jeter2019ai`.

`````

Verrebbe da pensare che sia facile, per una ragione già vista nella
{doc}`sezione su DQN <dqn>`: il Q-learning è *off-policy*, cioè il suo
bersaglio stima il valore della policy greedy mentre i dati possono venire da
un'altra policy. Se sa fare questo, dovrebbe saper imparare anche da partite
giocate da altri e finite da un pezzo: basterebbe addestrarlo sull'archivio
$\mathcal{D}$ invece che sulle esperienze fresche.

Fatto così, però, il metodo può fallire gravemente. Fujimoto e colleghi lo
mostrano su un compito di controllo continuo, perfino quando l'archivio è
generato da un agente che impara con successo da quelle stesse esperienze
{cite}`fujimoto2019off`. Con archivi enormi e vari, invece, come i cinquanta
milioni di transizioni accumulati da un DQN in tutto il suo addestramento su un
gioco Atari, gli stessi metodi possono superare, sulla maggior parte dei
giochi, l'agente che li ha prodotti {cite}`agarwal2020optimistic`. Capire
*quando* e *perché* l'archivio fa fallire il metodo è il cuore di tutto
l'argomento.

## Il buco nero delle azioni mai viste

Il colpevole è la differenza fra le azioni presenti nell'archivio e quelle che
l'agente, imparando, vorrebbe compiere. È la stessa frattura vista
nell'imitazione, quando l'allievo usciva dalla fascia dimostrata, e ha un nome
tecnico: **distributional shift**, cioè spostamento di distribuzione.

`````{tab} Elementare

Torniamo ai quaderni della nonna. Immagina che nei suoi decenni ai fornelli
non abbia *mai* messo il peperoncino in un piatto. Nel diario, quindi, alla
voce «peperoncino» non c'è nulla. Se ora ti chiedessero «quanto sarebbe buono
questo risotto con tre cucchiai di peperoncino?», la risposta onesta è «non ne
ho idea».

Un modello ingenuo, però, quando deve indovinare tira comunque un numero, e a
volte, per puro caso della sua matematica interna, tira un numero
*altissimo*. «Risotto al peperoncino: buonissimo, dieci e lode.»

L'agente sceglie sempre l'azione col voto più alto. Se un voto gonfiato per
sbaglio capita su un'azione mai provata, ci si butta a capofitto. Poiché non
può cucinarla per scoprire che è immangiabile, il voto sballato non viene mai
corretto, e per di più contagia le stime vicine. Nel gioco vero, dove le azioni
si provano, l'errore viene smascherato; qui resta e fa danni.

`````

`````{tab} Superiore

Consideriamo il target del Q-learning,
$y = r + \gamma \max_{a'} Q_\theta(s', a')$. Il problema è l'operatore
$\max_{a'}$, che spazia su *tutte* le azioni, comprese quelle che $\pi_\beta$
non ha mai eseguito in $s'$. Su quelle azioni **out-of-distribution** (OOD) la
rete $Q_\theta$ non ha mai visto dati e non fa che estrapolare. I suoi errori
di estrapolazione sono casuali, ma il $\max$ non è casuale: seleziona
sistematicamente i valori più alti, cioè le sovrastime. Di conseguenza il
target risulta gonfiato, la policy insegue quelle azioni OOD e, per
bootstrapping, il valore inflazionato si propaga all'indietro agli altri stati.

Nel RL online questo circolo si interrompe, perché eseguendo l'azione
sopravvalutata si osserva la ricompensa vera, bassa, e la stima si corregge.
Offline il correttivo non arriva mai (non c'è nuova interazione), per cui
l'errore può crescere senza freni.

Fujimoto e colleghi hanno mostrato che questo **extrapolation error** è la
ragione per cui gli algoritmi off-policy standard possono crollare su dati
fissi {cite}`fujimoto2019off`. Ad esempio, DDPG addestrato su un archivio di un
milione di transizioni resta molto sotto l'agente che lo ha raccolto, anche
quando i due imparano dagli stessi dati, e le sue stime oscillano o divergono.
Il limite, però, non è inevitabile. Agarwal e colleghi mostrano che su archivi
grandi e vari (tutta l'esperienza di un DQN, cinquanta milioni di transizioni
per gioco) QR-DQN e REM, due varianti di DQN che stimano rispettivamente una
distribuzione di ritorni e un insieme di teste di valore, addestrati offline
superano il DQN completamente addestrato sulla maggior parte dei giochi Atari
{cite}`agarwal2020optimistic`.

In generale, quanto conta l'estrapolazione dipende da quanto l'archivio copre
le azioni che la policy appresa vorrà fare. I rimedi hanno tutti lo stesso
obiettivo, cioè non fidarsi delle azioni fuori dal supporto dei dati.

`````

Prima dei rimedi, il fenomeno si vede in miniatura. L'azione è un numero solo,
da $-1$ a $+1$: la posizione dello sterzo. Il voto vero di ogni posizione lo
conosciamo noi, perché lo costruiamo come una multa: misura quanto ci si
allontana dalla posizione perfetta, col segno cambiato. La
posizione perfetta è $+0{,}3$, e lì la multa è nulla e il voto vale $0$; ogni
altra posizione paga qualcosa, tanto più quanto più è lontana. Zero è il
massimo, insomma, e tutti gli altri voti sono numeri negativi.

Chi ha raccolto i dati, però, è stato prudente: ha provato soltanto la fetta
fra $-1{,}0$ e $-0{,}2$, quaranta volte in tutto. L'azione migliore,
nell'archivio, non c'è. Il programma fa passare una curva per quei quaranta
punti e con quella dà un voto a tutte le azioni, comprese quelle mai provate.

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

Dall'uscita si vede che il voto più alto cade sullo sterzo tutto a destra, dove
nessuno ha mai messo piede: lì la curva promette $+30{,}1$, e il valore vero è
$-0{,}49$, mentre il voto migliore possibile vale zero.

```{figure} ../figures/supporto-dei-dati.svg
:name: fig-supporto-dei-dati
:alt: "Grafico del voto in funzione della posizione dello sterzo, da meno uno a più uno. Una fascia ocra copre il tratto da −1,00 a −0,20, il supporto dei dati, e dentro ci sono i quaranta punti osservati. La curva teal del voto vero è una parabola che ha il suo massimo, zero, in +0,30, cerchiato perché quella mossa nell’archivio non c’è. La curva terracotta stimata sui quaranta punti li segue dentro la fascia, e appena ne esce impenna e lascia il riquadro, salendo fino a +30,1 in più uno, dove il voto vero è −0,49. Cercando il massimo dentro la fascia si sceglie invece −0,21, dove stima −0,25 e valore vero −0,26 coincidono."
:width: 92%

Dentro la fascia ocra, dove si trovano i quaranta punti, la curva stimata e
quella vera si sovrappongono; appena fuori la stimata impenna e lascia il riquadro.
Cercare il voto più alto su tutto l'asse porta allo sterzo tutto a destra, con
una promessa di $+30{,}1$ a fronte di un valore vero di $-0{,}49$; cercarlo
dentro la fascia porta a $-0{,}21$, dove promessa e valore coincidono.
```

Non c'è niente di misterioso, e {numref}`fig-supporto-dei-dati` fa vedere dove
succede. Infatti una curva passa docilmente in mezzo ai quaranta punti che ha,
e fuori da quel tratto prosegue come le pare: non ha più nessun dato che la
trattenga, e le basta una piccola pendenza sbagliata al bordo per schizzare in
alto nel giro di un intervallo. La flessibilità decide quanto forte è lo
schizzo, non se ci sarà. Rifacendo, ad esempio, lo stesso calcolo con una retta
al posto della curva, il voto più alto cade di nuovo sullo sterzo tutto a
destra, con una promessa di $+2{,}06$ dove il migliore possibile vale zero. Si
sarebbe al riparo soltanto scegliendo la forma giusta, che qui è la parabola;
ma la forma giusta è quella che non si conosce, perché è ciò che si sta
cercando.

Vincolando invece la ricerca alla zona che i dati coprono (il loro
**supporto**, che con quaranta punti è il tratto che li racchiude) si sceglie
$-0{,}21$, e stima ($-0{,}25$) e realtà ($-0{,}26$) tornano a coincidere. È la
prima famiglia di soluzioni in miniatura, e ne mostra il limite: l'azione
perfetta, $+0{,}3$, resta irraggiungibile, perché nell'archivio non c'è. Si
rinuncia al meglio per non prendere abbagli, e l'RL offline accetta questo
compromesso di continuo.

## BCQ: restare vicini a ciò che è stato visto

L'idea, suggerita dall'esempio, è semplice. Se il problema nasce
dall'attribuire un voto ad azioni mai provate, conviene escluderle, cioè
cercare il voto più alto soltanto fra le azioni che compaiono nell'archivio.

A farlo nel controllo continuo, con le reti profonde e per dati fissi, è
**BCQ** (*Batch-Constrained deep Q-learning*) di Scott Fujimoto, David Meger e
Doina Precup {cite}`fujimoto2019off`, presentato nel 2019 e indicato dagli
autori stessi come il primo algoritmo di questo tipo.

`````{tab} Elementare

Per sapere quali ricette siano plausibili, BCQ impara a imitare chi ha
raccolto i dati. Da una parte addestra una rete a rispondere alla domanda «in
una serata come questa, che cosa avrebbe cucinato la nonna?», e quella rete
propone una manciata di ricette possibili, tutte del genere che compare nei
quaderni. Dall'altra parte la rete dei voti giudica soltanto quelle,
e si tiene la migliore.

La differenza è tutta qui: alle ricette che nessuno ha mai scritto non viene mai
chiesto un voto, quindi nessun voto di fantasia può vincere, perché non è mai
stato dato.

BCQ non può allora fare altro che ripetere la nonna? No, e per due motivi. Il
primo è che fra le ricette plausibili può scegliere la *migliore*, mentre la
nonna a volte sbagliava; già questo basta a fare meglio di lei. Il secondo è
che a ciascuna ricetta proposta è concesso un ritocco piccolo: una rete impara
quanto ritoccare, ma fin dove le sia concesso spingersi è un numero che si
sceglie a mano, prima di cominciare, e alzarlo o abbassarlo sposta BCQ fra il
copiare la nonna e il fidarsi dei voti. Il ritocco è un rischio, certo, ma di
taglia controllata: si resta accanto a qualcosa che è stato davvero cucinato, e
non si inventa un piatto in mezzo al nulla.

`````

`````{tab} Superiore

Per sapere quali azioni siano plausibili, BCQ addestra un modello generativo
(un *variational autoencoder*: un'architettura che comprime i dati in poche
variabili latenti e da quelle li ricostruisce, derivata nella {doc}`sezione sul
salto probabilistico </ModelliLatenti/il-salto-probabilistico>`) sulle coppie
$(s, a)$ del dataset: dato uno stato, genera azioni simili a quelle che
$\pi_\beta$ avrebbe scelto in situazioni analoghe. La funzione $Q$ viene poi
massimizzata solo su un piccolo insieme di azioni campionate da questo
generatore, ciascuna con una piccola perturbazione appresa che concede un
margine di miglioramento:

$$
\pi(s) = \operatorname*{arg\,max}_{a_i + \xi_\phi(s, a_i, \Phi)}
Q_\theta\big(s,\, a_i + \xi_\phi(s, a_i, \Phi)\big),
\qquad a_i \sim G_\omega(s),\quad i = 1, \dots, n,
$$

dove $G_\omega$ è il generatore, $\xi_\phi$ la perturbazione, che non può
superare in ampiezza $\Phi$, e $n$ il numero di azioni candidate (nel lavoro $n
= 10$). I due parametri spostano BCQ fra due estremi: con $\Phi = 0$ e $n = 1$
si ricade nella clonazione comportamentale, con $\Phi$ pari all'intera ampiezza
delle azioni e $n \to \infty$ torna il Q-learning su tutto lo spazio.

Lo stesso insieme di candidati serve a costruire il bersaglio, e in questo modo
la propagazione dell'errore si interrompe. In particolare, con $a'_i \sim
G_\omega(s')$, $\tilde a_i = a'_i + \xi_\phi(s', a'_i, \Phi)$ e due reti
$Q_{\theta_1}$, $Q_{\theta_2}$ (con le rispettive copie ritardate $\theta'_j$),

$$
y = r + \gamma \max_{i}\Big[\lambda \min_{j=1,2} Q_{\theta'_j}(s', \tilde a_i)
+ (1-\lambda)\max_{j=1,2} Q_{\theta'_j}(s', \tilde a_i)\Big],
$$

dove $\lambda$ regola quanto pesa l'incertezza sui passi futuri (nel lavoro
$\lambda = 0{,}75$; con $\lambda = 1$ è il *clipped double Q-learning*).
L'operatore di massimizzazione non può più cadere nelle
regioni OOD, perché sceglie il meglio *tra ciò che si sarebbe davvero potuto
fare*.

`````

È il modo più letterale di rispondere al distributional shift: costruire un
recinto attorno ai dati e non uscirne.

## CQL: essere pessimisti sull'ignoto

BCQ mette un recinto attorno alle azioni ammesse, e per farlo gli serve una
seconda rete, quella che imita chi ha raccolto i dati. Si può ottenere lo
stesso effetto senza quella seconda rete, e nel 2020 Aviral Kumar, Aurick Zhou,
George Tucker e Sergey Levine mostrano come: **CQL** (*Conservative
Q-Learning*) {cite}`kumar2020conservative`. Invece di vietare le azioni mai
viste (le azioni fuori distribuzione, o OOD), le rende *poco appetibili*,
intervenendo direttamente sulla *loss* dell'addestramento.

`````{tab} Elementare

Un critico gastronomico prudente si dà una regola: dei piatti documentati nei
quaderni si fida di ciò che c'è scritto, di quelli che non ha mai assaggiato dà
per scontato il peggio. Il pessimismo non è uniforme: va soprattutto ai piatti
mai provati che sulla carta sembrano più promettenti, perché sono quelli che
rischiano di trascinarlo in una serata storta.

CQL insegna questa prudenza alla rete dei voti. Una rete non impara in un colpo
solo: si corregge migliaia di volte, un pochino per volta, e a ogni correzione
CQL aggiunge due spinte. Abbassa i voti delle ricette che nel diario non
compaiono, e alza quelli delle ricette che vi figurano. Ne esce un sistema di
voti *conservativo*, pessimista su tutto ciò che non ha visto.

Quanto forte spinge è una manopola (in gergo, un iperparametro), e la si gira
a mano. Girata
poco, la prudenza resta un'inclinazione: un piatto mai assaggiato che il
critico si figura strepitoso può ancora finire in cima alla lista. Girata a
fondo, tutto ciò che non è documentato sprofonda, e allora il critico dai
quaderni non esce più, nemmeno quando fuori ci sarebbe di meglio.

La promessa che si riesce a dimostrare, poi, è più modesta della regola. Prendi
una serata e i piatti che il critico ordinerebbe: la media dei voti che dà a
quei piatti non supera ciò che la serata renderà, se si ordina come dice lui.
Un singolo piatto dentro il gruppo può ancora essere sopravvalutato, e può
ancora battere un piatto documentato. La garanzia è sulla media, non su ogni
riga della lista, e vale su una serata come quelle che i quaderni raccontano:
su una serata che nei quaderni non c'è non è promesso niente.

Vale a due condizioni. La prima è che la manopola sia girata abbastanza, e
quanto basti dipende da quanto ciascun piatto è documentato: uno cucinato una
volta sola dà un voto malfermo e chiede una spinta all'ingiù più robusta di uno
cucinato cento volte. La seconda riguarda il modo in cui il critico mette
insieme i piatti che si somigliano. Finché ogni piatto ha la sua scheda, o il
critico ha deciso le somiglianze una volta per tutte («i risotti con i risotti,
le zuppe con le zuppe»), abbassare il voto di un piatto immaginato non tocca
gli altri, e la promessa tiene. Un critico vero, però, impara strada facendo
che cosa somiglia a che cosa, e allora la spinta che abbassa un piatto
immaginato scivola anche su quelli che gli somigliano, compresi quelli scritti
nei quaderni: la prudenza continua a funzionare in pratica, ma la garanzia
teorica viene meno.

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
conservatività. Il primo termine riduce i valori $Q$ delle azioni campionate da
$\mu$ (tipicamente OOD), mentre il secondo li incrementa sulle azioni realmente
presenti in $\mathcal{D}$. Kumar e colleghi dimostrano che, per $\alpha$
abbastanza grande e con $\mu$ agganciata alla policy che si sta valutando, il
valore atteso delle azioni sotto la $Q$ così ottenuta minora il vero valore
della policy in ogni stato del dataset: un limite inferiore *in valore*, non
punto per punto (la singola stima $Q(s,a)$ può ancora eccedere quella vera), e
per gli stati mai visti non è garantito affatto.

La prudenza sull'ignoto poggia quindi su una garanzia formale, ma con limiti di
validità precisi. In particolare, l'enunciato principale riguarda il caso
tabellare, e vale per un $\alpha$ abbastanza grande rispetto all'errore di
campionamento, che dipende da quante volte la coppia $(s,a)$ compare nel
dataset. In appendice gli autori estendono il risultato agli approssimatori
lineari e alle reti nel regime del *neural tangent kernel*, cioè con le
caratteristiche di fatto congelate: due ipotesi che una rete addestrata
normalmente non soddisfa. L'analisi di CQL con reti profonde vere è lasciata a
lavori futuri; nella pratica il metodo opera come un'euristica efficace,
motivata dal risultato teorico nei casi ideali.

`````

## IQL: non guardare mai fuori dai dati

CQL valuta ancora le azioni OOD, salvo poi penalizzarle. Nel 2022 Ilya
Kostrikov, Ashvin Nair e Sergey Levine propongono **IQL** (*Implicit
Q-Learning*) {cite}`kostrikov2022offline`: costruire una strategia migliore di
quella che ha raccolto i dati *senza mai chiedere alla rete dei voti che voto
darebbe a un'azione che nell'archivio non c'è*. Se non
guardi mai fuori, non puoi essere ingannato da ciò che c'è fuori.

Il nome dice che il miglioramento della policy resta implicito. Finora, per
sapere quanto vale una situazione, si prendeva il voto dell'azione migliore fra
tutte quelle possibili, e da lì entravano le azioni mai provate. IQL non fa mai
quel confronto: ottiene il valore per un'altra strada, come sottoprodotto di
una stima condotta sui soli dati, e l'azione fantasma non viene nemmeno
nominata.

`````{tab} Elementare

Come si fa a scegliere bene senza mai considerare piatti mai cucinati? IQL
cambia la domanda che rivolge ai quaderni. Non chiede più «quanto varrebbe
questa ricetta ipotetica?», che è la domanda da cui nascono i voti di fantasia:
chiede «nelle serate come questa, fin dove sono arrivate le ricette
provate, contando di più quelle andate meglio?», e quella risposta è il metro
della situazione. È come giudicare il potenziale di una cucina dai suoi piatti
più riusciti, senza fantasticare su menù mai esistiti.

I due numeri si calcolano uno dall'altro. Il voto di una ricetta tiene conto del
metro della situazione in cui ti lascia; il metro di una situazione esce dai
voti delle ricette provate lì. Aggiornarli insieme, nello stesso istante, dà due
conti che si rincorrono e non si posano più. Il rimedio è tenere sul tavolo una
copia dei voti stampata poco prima, come la copia congelata di DQN, e
ricalcolare il metro leggendo quella: serve qualcosa di fermo a cui appoggiarsi.

Con quel metro («quanto si ottiene qui scegliendo bene fra ciò che è stato
fatto») si rileggono poi le pagine del diario: le ricette che hanno reso più
del solito nella loro situazione vengono imitate di più, le altre di meno.
Nessuna azione fuori dal diario viene mai nemmeno nominata: dove gli altri
metodi mettono recinti o penalità, IQL elimina l'occasione di sbagliare.

`````

`````{tab} Superiore

L'idea consiste nel sostituire il $\max_{a'} Q(s',a')$ (che costringe a spaziare
su tutte le azioni) con una funzione valore $V(s')$ stimata tramite
**regressione expectile**. Un expectile alto (vicino a $1$) fa sì che
$V(s)$ approssimi il *massimo* dei valori $Q(s,a)$ ma solo sulle azioni $a$
*effettivamente osservate* in quello stato: coglie «il meglio tra ciò che è
stato fatto» senza mai nominare un'azione ipotetica. Con i valori di $\tau$ che
si usano in pratica la stima sta fra la media e il massimo, tanto più vicina al
massimo quanto più $\tau$ si avvicina a $1$. Le reti del valore si addestrano
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

Qui $\tau \in (0,1)$ è l’expectile (in pratica $\tau \approx 0{,}7$–$0{,}9$;
questo $\tau$ è il livello dell'asimmetria, e non ha niente a che vedere né con
la traiettoria né con il peso dello scorrimento di DDPG): la perdita
asimmetrica $L_2^\tau$ pesa di più i residui positivi, spingendo $V$ verso
l'alto della distribuzione dei $Q$ nel dataset. Il $Q$ dentro $\mathcal{L}_V$ è
una copia ritardata, come la rete-target di DQN: senza, le due regressioni si
inseguirebbero a vicenda senza niente di fermo a cui appoggiarsi. Il target di
$\mathcal{L}_Q$ usa $V(s')$, non un massimo su azioni arbitrarie: ecco perché
nessuna azione OOD viene mai valutata. La policy si estrae infine per
*advantage-weighted regression*, imitando le azioni del dataset con peso
$\exp\!\big(\beta_{\text{AWR}}\,(Q(s,a)-V(s))\big)$: l'esponenziale tiene i
pesi positivi anche sui vantaggi negativi, e $\beta_{\text{AWR}}$, l'inverso di
una temperatura (che con la policy comportamentale $\pi_\beta$ ha in comune
solo la lettera), decide quanto la policy assomigli a quella che ha raccolto i
dati ($\beta_{\text{AWR}} \to 0$) o a quella che insegue il vantaggio massimo
($\beta_{\text{AWR}}$ grande). Nell'implementazione degli autori i pesi sono
troncati a $100$, perché l'esponenziale non faccia esplodere un singolo
campione.

`````

Il nucleo di IQL, in PyTorch, sta in poche righe. È una misura
d'errore volutamente asimmetrica: con `tau=0.8` una stima per difetto viene
penalizzata quattro volte più di una stima per eccesso ($0{,}8$ contro
$0{,}2$). Poiché l'errore per difetto ha un peso quadruplo, il valore stimato
si colloca sopra la media dei voti osservati, pur senza raggiungere il
migliore: con valori distribuiti uniformemente fra $0$ e $1$ la stima si
attesta a $0{,}667$, e si avvicina al massimo solo quando $\tau$ tende a $1$.

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

Fin qui il Q-learning è stato adattato per sopravvivere ai dati fissi. Nel 2021
un gruppo di Berkeley, Facebook AI Research e Google Brain (Lili Chen, Kevin Lu
e colleghi) propone di cambiare prospettiva {cite}`chen2021decision`: trattare
l'apprendimento offline come un problema di modellazione di sequenze, lo stesso
compito su cui eccellono i Transformer alla base dei modelli linguistici.

`````{tab} Elementare

Un modello linguistico impara a completare le frasi: gli dài l'inizio e lui
predice la parola dopo, avendo letto miliardi di testi. Il **Decision
Transformer** fa lo stesso, ma la «frase» è una partita: una sequenza di
situazioni e mosse. Lo alleni sui diari delle partite passate a completare la
sequenza: «dopo questa situazione, che mossa è stata fatta?».

In più, all'inizio di ogni mossa gli si dice *quanto punteggio si vuole ancora
totalizzare da qui alla fine*. In addestramento questo numero è noto: basta
sommare le ricompense future. Al momento di giocare gli si dà un obiettivo
ambizioso («da qui voglio fare cento punti») e il modello, per coerenza con
tutte le partite viste, produce le mosse che *tipicamente portano a cento
punti*. Il traguardo si aggiorna a ogni mossa: incassati venti punti, da lì in
avanti la richiesta diventa ottanta. Il modello non calcola alcun valore:
racconta la partita che vorresti, e la recita.

Gli manca la capacità di rimediare. Se una mossa storta lo porta in una
posizione che nei diari non compare, produce comunque una mossa, ma risponde a
una domanda che non ha mai visto; e siccome la posizione successiva dipende da
quella risposta, lo scarto si allarga a ogni passo invece di rientrare.

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

In fase di controllo si fissa un return-to-go iniziale $\hat{R}_1$ desiderato,
si osserva lo stato, si genera l'azione; a ogni passo si decrementa il
return-to-go della ricompensa incassata e si prosegue. È il ponte esplicito tra
reinforcement learning e *sequence modeling*: condizionare sul risultato
voluto, anziché inseguire un valore stimato. Un confronto successivo suggerisce
che a contare sia soprattutto il condizionamento, più dell'architettura: sui
compiti di locomozione simulata una semplice rete densa condizionata sul
ritorno eguaglia il Decision Transformer {cite}`emmons2022rvs`.

`````

Il Decision Transformer non è sempre il migliore, e il caso in cui perde si
capisce: se nell'archivio non c'è una sola partita andata a finire bene,
ha ben poco da recitare, perché riproduce per lo più partite che ha visto. Gli
autori osservano che può chiedere anche più del massimo registrato, ma lo
presentano come qualcosa che riesce soltanto in alcuni giochi
{cite}`chen2021decision`.

CQL e IQL stimano invece i voti, e questo dà loro un potere in più: sanno
«ricucire». Un voto sta attaccato a una singola mossa in una singola
situazione, non a una partita intera; quindi se due partite mediocri passano
tutte e due per la stessa situazione, l'inizio buono della prima si può
attaccare al finale buono della seconda, e ne esce un percorso migliore di
entrambe, che nell'archivio non c'è. Il Decision Transformer, che racconta le
partite per intero, non ha un meccanismo per questa ricucitura. Gli autori la
osservano comunque su un piccolo problema di percorsi in un grafo, dove il
15,8% dei percorsi generati non compare in nessuna partita dell'archivio
{cite}`chen2021decision`; ma lì è un effetto osservato, mentre nei metodi a
valori discende dalla forma stessa del voto.

Il Decision Transformer ha comunque aperto, insieme al Trajectory Transformer di
Janner e colleghi, uscito il giorno dopo {cite}`janner2021offline`, una linea
feconda: una parte del reinforcement learning si può riformulare come
apprendimento supervisionato di sequenze, condizionato sul risultato che si
vuole.

## Un filo che torna: le preferenze dell'RLHF

Un altro filo porta all’RLHF, già introdotto nella {doc}`sezione sulla ricerca
ad albero <mcts-alphago>`, il modo in cui si addestrano gli assistenti
conversazionali {cite}`ouyang2022training`: delle persone confrontano a due a
due le risposte del modello e dicono quale preferiscono; da quei confronti si
costruisce un giudice automatico, il *modello di ricompensa*, che da lì in poi
assegna i voti al posto loro.

Quei confronti formano un archivio, che resta fermo mentre l'addestramento va
avanti. Ouyang e colleghi scrivono però che i due passi (raccogliere confronti
e ottimizzare) si possono iterare di continuo, tornando dalle persone a
chiedere nuovi giudizi sulle risposte della versione migliore del momento. Si
raccoglie dunque a tornate, e fra una tornata e l'altra il giudice resta
identico, mentre il grosso dei confronti su cui è stato costruito viene
da risposte di versioni precedenti.

Tanto basta perché l'RLHF erediti le tensioni dell'apprendimento offline. Il
modello si allontana dalle risposte che le persone hanno giudicato, e
là dove nessuno ha giudicato niente il giudice automatico è più facile da
imbrogliare: è di nuovo il problema delle azioni fuori dai dati. Il rimedio è
della stessa famiglia: al punteggio che il modello cerca di far salire si
sottrae una penalità che cresce con la distanza dalle risposte di partenza,
così che allontanarsi resta possibile ma sempre meno conveniente.

Questa dinamica spiega *perché* si itera. Ottimizzando, il modello si
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
  enormi e vari a volte funziona). L'agente cerca sempre il voto più alto, e i
  voti più alti finiscono per capitare sulle azioni mai provate, quelle su cui
  il modello può solo tirare a indovinare (il risotto al peperoncino da dieci e
  lode). Nessuna prova può smentire quel voto gonfiato, che anzi contagia le
  stime vicine.
- BCQ costruisce un recinto: valuta solo le azioni plausibili secondo
  l'archivio, generate imitando chi ha raccolto i dati. CQL non vieta
  nulla, insegna prudenza: abbassa i voti di ciò che non è mai stato provato e
  alza quelli di ciò che è documentato. La garanzia che se ne ricava è sulla
  media delle azioni che poi sceglierà, non su ogni singolo voto: uno gonfiato
  può ancora scapparci, e nelle situazioni che l'archivio non contiene non è
  promesso niente. IQL elimina l'occasione di sbagliare: chiede
  solo quanto hanno reso, in situazioni come questa, le azioni migliori fra
  quelle fatte, e non nomina mai un'azione fuori dal diario.
- Il Decision Transformer cambia domanda: tratta la partita come una frase
  da completare e si allena sui diari a predire la mossa successiva. Gli si
  dice anche quanto punteggio si vuole ancora totalizzare, e lui produce le
  mosse che di solito portano lì. Nessun voto da stimare, quindi nessun voto
  gonfiato.
- Anche i confronti con cui le persone giudicano le risposte di un assistente
  conversazionale sono un archivio, che si riapre a tornate ma resta fermo
  mentre l'addestramento va avanti: il giudice costruito su quei confronti si
  inganna più facilmente là dove nessuno ha giudicato niente, e il rimedio
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
  condiziona sul *return-to-go* desiderato e predice l'azione con un
  Transformer, in modo puramente supervisionato {cite}`chen2021decision`.
- I dati di preferenza dell’RLHF sono anch'essi un dataset raccolto a
  tornate e fermo fra una tornata e l'altra: il modello di ricompensa viene
  interrogato fuori dal suo supporto e si lascia sfruttare
  {cite}`gao2023scaling`, e la penalità KL verso il modello di partenza è un
  rimedio della stessa famiglia {cite}`ouyang2022training`.
```
`````
