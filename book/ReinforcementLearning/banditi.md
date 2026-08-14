# Il problema più semplice: i bandit a più braccia

Nel 1933, sulle pagine della rivista *Biometrika*, William R. Thompson pone una
domanda che nasce da un disagio pratico {cite}`thompson1933likelihood`. In una
sperimentazione clinica si assegnano i pazienti a due trattamenti e si aspetta
la fine per sapere quale funzioni meglio. Ma a metà strada un'idea di quale sia
il migliore già ce l'abbiamo: continuare ad assegnare metà dei pazienti al
trattamento che sta perdendo è il prezzo che si paga per essere sicuri.
Thompson si chiede se quel prezzo si possa ridurre spostando via via
l'assegnazione verso il trattamento che sta andando meglio, senza per questo
smettere di raccogliere prove sull'altro.

Una risposta la diede lui stesso, e vale la pena anticiparla perché la storia
ha un finale: assegnare ogni paziente al trattamento con la probabilità che
quel trattamento sia davvero il migliore, viste le prove raccolte fino a quel
momento. Chi va meglio riceve più pazienti, ma nessuno viene scartato finché
resta un dubbio. L'idea rimase quasi ignorata per decenni e porta il nome del
suo autore, *Thompson sampling*; è una delle ricette usate oggi nei test A/B
adattivi di cui si parla in fondo a questa sezione. Qui però prendiamo altre
tre strade, più semplici da raccontare e più facili da mettere in codice.

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

Hai dieci leve davanti. Ognuna, quando la tiri, ti dà una somma che cambia ogni
volta: alcune leve sono in media generose, altre in media avare, ma nessuna
è costante e tu non sai quali siano quali. Hai mille tiri.

Tutto ciò che puoi fare è tenere un quaderno: per ogni leva, la media di quanto
ti ha reso finora. Quella media è la tua **stima**. All'inizio è pessima
perché si basa su un tiro o due; con l'uso migliora.

Il quaderno si aggiorna senza rifare la somma da capo, con una regola che vale
la pena guardare in faccia perché ritorna dappertutto in questo libro:

> stima nuova = stima vecchia + passo × (quello che ho appena visto − stima vecchia)

Cioè: sposto la stima verso la sorpresa, di un tanto deciso dal passo.

Se il passo è "uno diviso il numero di volte che ho tirato questa leva", si
ottiene esattamente la media di tutti i tiri, e conviene vederlo su due numeri
invece di crederci sulla parola. Parto da $0$ e tiro due volte, incassando
prima $4$ e poi $6$. Primo tiro: il passo è $1/1 = 1$, la stima diventa
$0 + 1 \times (4 - 0) = 4$. Secondo tiro: il passo è $1/2$, la stima diventa
$4 + 0{,}5 \times (6 - 4) = 5$, che è precisamente la media fra $4$ e $6$.
Funziona così a ogni tiro: il passo che si accorcia è quello che tiene in
equilibrio le vecchie osservazioni con la nuova.

Se invece il passo lo tengo **fisso**, le osservazioni recenti pesano di più e
quelle vecchie svaniscono piano piano: è quello che serve se le leve cambiano
carattere nel tempo, cosa che nel mondo reale succede sempre.

`````

`````{tab} Superiore

Il problema ha $k$ azioni. Ogni azione $a$ ha un valore vero
$q_*(a) = \mathbb{E}[R_t \mid A_t = a]$, ignoto, e la ricompensa osservata è
una realizzazione rumorosa attorno a quel valore. Non c'è stato: la
distribuzione delle ricompense non dipende da cosa è successo prima. Un bandit
è, se si vuole, un MDP con un solo stato.

Un avvertimento sugli indici, perché è una trappola classica e cade proprio
qui. In questa sezione la ricompensa dell'azione $A_t$ si indicizza $R_t$,
perché non c'è uno stato successivo di cui tenere il passo; dalla prossima
sezione in poi, dove lo stato c'è, la stessa ricompensa si scriverà $R_{t+1}$,
come nella panoramica. È la convenzione di Sutton e Barto ed è comoda da
entrambe le parti, ma va tenuta a mente confrontando le formule di qui con
quelle di là.

La stima naturale di $q_*(a)$ è la **media campionaria**

$$
Q_t(a) = \frac{\sum_{i<t} R_i \cdot \mathbb{1}[A_i = a]}{\sum_{i<t} \mathbb{1}[A_i = a]},
$$

che si calcola in forma incrementale, senza tenere in memoria la storia. Se
$Q_n$ è la stima dopo $n-1$ tiri della stessa leva e $R_n$ è l'$n$-esima
ricompensa,

$$
Q_{n+1} = Q_n + \frac{1}{n}\big(R_n - Q_n\big).
$$

È la forma canonica di ogni regola di apprendimento di questo libro:
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

Il criterio che distingue i due casi vale la pena scriverlo, perché è lo stesso
che tornerà a chiedere la garanzia di convergenza del Q-learning. Una successione
di passi $\alpha_n$ porta la stima al valore vero se soddisfa le **condizioni di
Robbins-Monro**

$$
\sum_{n=1}^{\infty} \alpha_n = \infty,
\qquad
\sum_{n=1}^{\infty} \alpha_n^2 < \infty :
$$

la prima chiede che i passi restino abbastanza grandi da poter raggiungere
qualunque punto di partenza, la seconda che si accorcino abbastanza in fretta da
smettere di rincorrere il rumore. Il passo $1/n$ le soddisfa entrambe
($\sum 1/n$ diverge, $\sum 1/n^2$ converge); un passo costante $\alpha$ soddisfa
la prima e viola la seconda, ed è per questo che non converge. Non è un difetto:
è la proprietà che lo rende adatto ai problemi non stazionari.

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
classico della materia. Il banco è il loro; le percentuali di questa sezione
no, escono dal codice in fondo, che lo rifà da capo.

`````{tab} Elementare

Il banco è fatto così. Dieci leve, mille tiri in tutto. Ogni leva ha un suo
**valore vero**, cioè quanto rende in media, e questo valore viene sorteggiato
attorno allo zero: qualche leva rende un po' più di zero, qualcuna un po' meno,
nessuna moltissimo. Quando tiri una leva incassi il suo valore vero più un
errore casuale, di solito non più grande di uno: ecco perché due tiri della
stessa leva danno numeri diversi, ed ecco perché le tue stime, all'inizio, non
valgono niente. E poiché con dieci leve sorteggiate una volta sola si rischia di
essere fortunati o sfortunati per caso, l'esperimento intero si rifà da capo
duemila volte con leve nuove e si fa la media dei risultati.

Il punteggio con cui si giudica una strategia è la percentuale di volte in cui,
**negli ultimi cento tiri**, sta tirando davvero la leva migliore. Ultimi cento
e non tutti e mille, perché all'inizio nessuno può saperlo: quel che interessa è
che cosa ha imparato alla fine.

Un avvertimento per non prendere troppo alla lettera la macchinetta del casinò:
qui quello che si incassa non sono soldi, è un punteggio che può benissimo
essere negativo, e non esiste la scelta di non giocare. La domanda non è se
convenga tirare, è **quale** leva tirare.

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

Su quel banco l'agente avido sceglie la leva migliore solo nel **36,7%** dei
casi: dopo mille tentativi, due volte su tre sta ancora tirando la leva
sbagliata.

Basta pochissimo per cambiare le cose. Con $\varepsilon$-greedy, cioè una leva
a caso una volta ogni dieci, si sale all'**80,2%**. La sua virtù non è di non
avere parametri, perché $\varepsilon$ è un parametro a tutti gli effetti e
poche pagine più avanti vedremo che va scelto guardando quanto dura la partita;
la sua virtù è di essere **robusta**: sbagliare $\varepsilon$ di un fattore
dieci costa molto meno che non esplorare affatto. Il suo difetto, però, è
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
al **71,3%**, quasi il doppio del 36,7% dell'agente avido; ma quasi un terzo
del guadagno se n'è andato, perché con il passo fisso si sale all'**86,6%**, il
risultato migliore fra le strategie di questa sezione.

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
69% del divario fra l'avido e la migliore strategia della sezione. Il termine di
paragone corretto per l'ottimismo è quello, non $\varepsilon$-greedy; resta però
che con la media campionaria il risultato finisce sotto l'80,2% della leva a
caso una volta ogni dieci, mentre con $\alpha = 0{,}1$ costante l'ottimismo si
consuma abbastanza lentamente da arrivare all'**86,6%**.

`````

È un trucco, però, e conviene dire perché. L'ottimismo si esaurisce: dopo che
tutte le leve sono state provate abbastanza, la spinta a esplorare sparisce.
Se le leve rendono sempre allo stesso modo (si dice che il problema è
**stazionario**) va benissimo; se invece cambiano carattere nel tempo, e
servirebbe tornare a esplorare perché quel che si era imparato non vale più,
non serve a niente. Come scrivono Sutton e Barto, l'inizio del tempo capita una
volta sola, e non conviene puntarci troppo.

### UCB: esplorare in proporzione a quanto poco si sa

Le tre lettere stanno per *upper confidence bound*, "estremo superiore
dell'intervallo di confidenza", e il nome dice già il metodo: il numero su cui
si decide non è la stima di una leva, ma il valore più alto che quella leva
potrebbe ancora avere viste le prove raccolte finora.

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

Il bonus cresce anche col passare del tempo, e non è un dettaglio: significa
che una leva trascurata a lungo torna prima o poi in cima alla lista. Nessuna
leva viene abbandonata per sempre, ma le peggiori vengono ricontrollate sempre
più di rado.

`````

`````{tab} Superiore

L'**Upper Confidence Bound** sceglie

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
limitate, un'ipotesi che il banco di prova gaussiano di queste pagine, a
rigore, non rispetta.

Per confronto, $\varepsilon$-greedy con $\varepsilon$ costante ha rimpianto
**lineare** in $T$, perché continua a sbagliare una frazione fissa delle volte
per sempre: è la differenza fra un'esplorazione che si dosa e una che non si
spegne mai.

Sul banco di prova, con $c = 2$, UCB sceglie la leva migliore l'**85,9%** delle
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

Il **gradiente** che dà il nome al metodo è quello del capitolo di matematica,
la direzione lungo cui una quantità cresce più in fretta: qui i voti delle leve
non si stimano, si spingono a ogni tiro un poco nella direzione che fa salire la
ricompensa che ci si aspetta.

`````{tab} Elementare

Le strategie viste finora stimano *quanto vale* ogni leva e poi decidono. Se ne
può fare a meno: si può imparare direttamente una **preferenza**, un voto senza
unità di misura, e tirare ogni leva tanto più spesso quanto più alto è il suo
voto.

Su come si passa dai voti alle probabilità c'è però una cosa da mettere subito
in chiaro, perché è tutto il punto del paragrafo che segue: non conta il voto in
sé, conta **quanto è più alto degli altri**. Alzare tutti i voti della stessa
quantità non cambia niente, come in una classifica a punti: se do un punto in
più a tutti, l'ordine e i distacchi restano identici.

La regola di aggiornamento è di buon senso: se la ricompensa appena incassata è
**migliore della media** di quelle ricevute finora, alzo il voto della leva che
ho tirato e abbasso quello di tutte le altre; se è peggiore, faccio l'opposto.

Quel confronto con la media è il pezzo importante e si chiama **termine di
riferimento** (in inglese *baseline*, ed è il nome che si legge nel codice).
Senza, l'algoritmo confronterebbe la ricompensa con lo zero, che è un numero
arbitrario: se tutte le leve pagano attorno a mille, tutte le ricompense
sembrano ottime e i voti salgono tutti insieme senza distinguere nulla. Con il
riferimento, quel che conta non è quanto ho preso, ma quanto ho preso
**rispetto al solito**.

Quanto conti si misura, ed è tanto. Sul banco di prova di prima il metodo dei
voti azzecca la leva migliore l'**84,1%** delle volte. Se poi si aggiungono
quattro punti a tutte le ricompense, cosa che non cambia nulla del problema
(fra le leve le differenze restano quelle), con il riferimento si resta
all'**83,8%** e senza si crolla al **48,5%**.

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

dove $\bar{R}_t$ è la media delle ricompense fino a $t$, cioè la **baseline**.

Vale la pena riconoscere che cosa si sta guardando: è **REINFORCE con
baseline**, il metodo a gradiente di policy del capitolo sul deep
reinforcement learning, nel caso degenere di un solo stato. La stessa
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
metodo arriva all'**84,1%**. Traslando **tutte** le ricompense di $+4$, cosa
che non cambia in nulla la difficoltà del problema (le differenze fra le leve
sono identiche), con la baseline si resta all'**83,8%**, mentre togliendola si
crolla al **48,5%**. La baseline non è un'ottimizzazione: è ciò che rende
l'algoritmo indifferente all'origine della scala delle ricompense.

`````

## Alla prova: duemila banchi da mille tiri

I numeri citati qui sopra non sono copiati da nessuno, sono usciti dal codice
che segue. Le quattro strategie basate sui valori (avida, $\varepsilon$-greedy,
ottimista, UCB) vivono in un solo blocco, perché differiscono solo per come
scelgono l'azione e per come iniziano; le sei righe stampate sono le stesse
quattro, con due valori di $\varepsilon$ e due modi di fare il passo.

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
dopo mille tiri sta al **59,1%**, contro l'**80,2%** di chi esplora una volta su
dieci, e sembra il peggiore dei rimedi. Ma sta ancora salendo. Esplorando una
volta su cento impiega dieci volte più tempo a farsi un'idea di tutte le leve, e
alla fine supera l'altro, che invece continuerà per sempre a buttare un tiro su
dieci: portando `PASSI` da mille a trentamila nel codice qui sopra, il sorpasso
si vede arrivare attorno al decimillesimo tiro, e alla fine i due valgono
**91,6%** e **89,0%**, con le parti invertite. La classifica dipende insomma da
quanto è lunga la partita, e questa è una morale generale: **quanto esplorare
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

## Dove si incontrano davvero

Un bandit non è un giocattolo teorico, ed è probabilmente la parte di
reinforcement learning che più spesso finisce in produzione, proprio perché
rinuncia a tutto il resto.

**Test A/B, e il loro superamento.** Un test A/B è la prova che si fa quando si
hanno due versioni di una pagina, di un annuncio o di un prezzo e si vuole
sapere quale rende di più: si mostra la prima a metà dei visitatori e la seconda
all'altra metà, fino alla fine dell'esperimento. È la sperimentazione clinica di
Thompson, con gli stessi costi. Chi la fa in modo adattivo sposta via via i
visitatori verso la versione che sta vincendo, e paga in difficoltà di lettura
statistica (i dati non sono più raccolti allo stesso modo per tutti) quello che
guadagna in denaro non buttato.

**Esplorazione nei sistemi di raccomandazione.** Un catalogo ha continuamente
oggetti nuovi, di cui nessuno sa nulla: mostrarli è esplorare, e non mostrarli
mai garantisce che nessuno saprà mai se erano buoni. È il problema che il
capitolo sui sistemi di raccomandazione, più avanti nel libro, chiamerà
**partenza a freddo**, qui letto dal lato della decisione invece che da quello
della rappresentazione.

**Ricerca di iperparametri.** Qui il collegamento è letterale. Il *successive
halving* e **Hyperband** del capitolo sul machine learning sono algoritmi di
bandit: ogni configurazione è una leva, addestrarla per un'epoca è un tiro, e
il problema si chiama *best-arm identification*, che è la variante in cui non
interessa massimizzare le ricompense lungo la strada ma solo indovinare alla
fine qual era la leva migliore.

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
- Due modi di spenderla meglio: partire da stime **troppo generose**, così che
  ogni leva deluda e l'agente le giri tutte da solo (86,6%, ma il trucco si
  consuma e non serve se il mondo cambia); oppure aggiungere a ogni stima un
  **bonus di ignoranza**, tanto più grande quanto meno si è provata quella leva
  (85,9%).
- Si può anche non stimare nulla e imparare direttamente dei **voti**, alzando
  quello della leva appena tirata se ha reso più del solito e abbassandolo se ha
  reso meno. Il confronto "rispetto al solito" non è un dettaglio: senza,
  aggiungendo quattro punti a tutte le ricompense il metodo crolla dall'84% al
  48%.
- Non sono giocattoli: si incontrano nei test A/B che spostano il traffico verso
  la variante che sta vincendo, nel decidere quali oggetti nuovi mostrare in un
  catalogo, e nella ricerca degli iperparametri (ogni configurazione è una leva,
  addestrarla un po' è un tiro). Il gradino successivo è il caso in cui prima di
  scegliere si guarda la situazione, ma le proprie scelte non la cambiano.
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
  attorno al decimillesimo tiro.
- **Valori iniziali ottimisti** (86,6%, con passo fisso; 71,3% con la media
  campionaria, contro il 36,7% dell'avido puro): esplorazione quasi gratis,
  ma si esaurisce e non serve sui problemi non stazionari. **UCB** (85,9%):
  esplora di più le leve di cui sa di meno, e quello che perde per farlo
  cresce con il ritmo più lento possibile per una strategia che debba
  funzionare su qualunque insieme di leve (Lai e Robbins).
- Il **bandit a gradiente** impara preferenze invece di valori, ed è REINFORCE
  con baseline in miniatura. La baseline non è un dettaglio: traslando le
  ricompense di $+4$, senza di essa si passa dall'84% al 48%.
- Si incontrano davvero in test A/B adattivi, esplorazione nei sistemi di
  raccomandazione e ricerca di iperparametri (Hyperband è *best-arm
  identification*). Il gradino successivo è il **bandit contestuale**, dove
  c'è uno stato ma le azioni non lo cambiano.
```

`````
