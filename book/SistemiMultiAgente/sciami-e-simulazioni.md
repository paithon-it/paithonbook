# Molti semplici invece di pochi intelligenti

Gli storni con cui si è aperto il capitolo non stanno risolvendo niente: volano.
La loro regola dei sei o sette vicini serve a restare insieme sopra il posatoio,
non a calcolare qualcosa. Ma la domanda che quello spettacolo mette in testa a
un informatico è vecchia e precisa: se una regola locale elementare basta a
tenere in aria migliaia di uccelli, può bastare anche a **risolvere un
problema**?

Per una trentina d'anni la risposta a questa domanda ha occupato una fetta
grossa della ricerca multi-agente, e conviene dirlo perché oggi lo si dimentica:
prima che «agente» significasse un modello di linguaggio con un prompt di
sistema, in buona parte di quella letteratura un agente era una particella con
tre righe di aritmetica dentro. Le sezioni precedenti hanno contato token e
tipizzato messaggi perché i partecipanti erano cari e loquaci; qui il regime si
ribalta. I partecipanti sono centinaia, costano niente, non
ragionano e non si parlano. L'idea comune resta però quella dell'apertura:
molte unità quasi banali, nessun controllore centrale, e una soluzione che
**emerge** dall'interazione invece di essere calcolata da qualcuno.

## Le formiche del Politecnico

L'idea nasce da una domanda di etologia: come fanno animali quasi ciechi come le
formiche a trovare il cammino più breve fra il formicaio e una fonte di cibo,
senza vederlo e senza che nessuna di loro lo sappia? La prima risposta esce nel
1991 a Parigi, in un articolo di convegno di Alberto Colorni, Marco Dorigo e
Vittorio Maniezzo alla First European Conference on Artificial Life. Diventa nel
1992 la tesi di dottorato di Dorigo al Politecnico di Milano, scritta in
italiano e da allora citata con il titolo inglese *Optimization, Learning and
Natural Algorithms*; e nel 1996 l'articolo di rivista che tutti citano, *Ant
system: optimization by a colony of cooperating agents*, sulle *IEEE
Transactions on Systems, Man, and Cybernetics, Part B* {cite}`dorigo1996ant`. È
il capostipite di una famiglia di algoritmi (la **ant colony optimization**)
nata in un'università italiana, e il problema su cui viene provato è il
**commesso viaggiatore**: date certe città e le distanze fra loro, trovare il
giro più corto che le tocchi tutte una volta sola e torni al punto di partenza.
È facile da enunciare e feroce da risolvere, perché i giri possibili crescono in
modo mostruoso col numero di città, ed è per questo uno dei problemi più
studiati che esistano.

Il meccanismo biologico è a una riga. Una formica che cammina deposita per
terra una sostanza, il **feromone**; una formica che incontra una traccia già
depositata tende a seguirla, e seguendola la rinforza con il proprio feromone.
Il resto è aritmetica.

`````{tab} Elementare

Mettiamo due strade che portano allo stesso cibo, una lunga il doppio
dell'altra, e formiche che vanno tutte alla stessa velocità. All'inizio nessuna
sa niente e si dividono a caso, metà di qua e metà di là.

Adesso guarda l'orologio. Se il giro breve si fa in dieci minuti e quello lungo
in venti, in un'ora una formica sulla strada corta la percorre sei volte e una
sulla strada lunga tre. Stesso numero di formiche, stessa quantità di feromone
lasciata a ogni passaggio, e sulla strada corta se ne accumula il doppio. Le
formiche che arrivano dopo trovano una traccia doppia da una parte, la seguono
con probabilità maggiore, e depositano altro feromone proprio lì.

Nessuno ha misurato le due strade. Nessuno le ha confrontate. La strada corta
vince perché la si percorre più spesso, e la si percorre più spesso perché è
corta: è il tempo a fare la misura al posto di un cervello.

`````

`````{tab} Superiore

La colonia costruisce soluzioni un pezzo alla volta. Una formica $k$ che si
trova sul nodo $i$ sceglie il nodo successivo $j$ con probabilità

$$
p_{ij}^{k} \;=\; \frac{\tau_{ij}^{\alpha}\;\eta_{ij}^{\beta}}
{\displaystyle\sum_{l \,\in\, \mathcal{A}_k} \tau_{il}^{\alpha}\;\eta_{il}^{\beta}},
\qquad j \in \mathcal{A}_k,
$$

dove $\tau_{ij}$ è la quantità di feromone sull'arco $(i,j)$, $\eta_{ij}$ è la
**visibilità**, cioè l'inverso della lunghezza dell'arco ($\eta_{ij} = 1/d_{ij}$
nel commesso viaggiatore, e niente a che vedere con il tasso di apprendimento
$\eta$ del resto del libro), $\mathcal{A}_k$ è l'insieme dei nodi che la formica
$k$ non ha ancora visitato (glielo impedisce una lista di nodi proibiti, che è
ciò che rende il giro legale) e gli esponenti $\alpha, \beta \ge 0$ pesano i due
termini l'uno contro l'altro. La formula è un compromesso fra **esperienza
collettiva** ($\tau$: quante formiche sono passate di qui e quanto bene è
andata) ed **euristica locale** ($\eta$: quanto è vicino il prossimo nodo). I
due casi limite lo chiariscono: con $\alpha = 0$ il feromone sparisce dal conto
e resta un algoritmo goloso stocastico, cioè una colonia di formiche che non
comunicano; con $\beta = 0$ resta solo il passaparola, senza nessuna
informazione sul problema.

Il rinforzo arriva a giro finito, ed è proporzionale alla **qualità** della
soluzione costruita:

$$
\Delta\tau_{ij} \;=\; \sum_{k=1}^{m} \Delta\tau_{ij}^{k},
\qquad
\Delta\tau_{ij}^{k} \;=\;
\begin{cases}
Q / L_k & \text{se la formica } k \text{ ha usato l'arco } (i,j),\\[2pt]
0 & \text{altrimenti,}
\end{cases}
$$

dove $L_k$ è la lunghezza del giro completo costruito dalla formica $k$ e $Q$ è
una costante di scala dell'algoritmo (nessun rapporto con il valore-azione
$Q$ del Reinforcement Learning; nel lavoro originale la sua influenza risulta
trascurabile). Il quoziente $Q/L_k$ è la differenza che conta fra questa
versione (*ant-cycle*) e le due che gli autori provano e scartano, dove la
formica deposita a ogni passo, senza aspettare la fine del giro, una quantità
fissa (*ant-density*) o inversamente proporzionale alla lunghezza del singolo
arco (*ant-quantity*). Sono due usi di informazione locale, e infatti vanno
peggio: chi ha fatto un giro corto deve lasciare più feromone di chi ne ha fatto
uno lungo, e per saperlo bisogna che il giro sia finito. Così la traccia non
registra il traffico, registra il **merito**.

`````

Questo è uno dei pochi meccanismi del libro in cui **il tempo è il contenuto**:
non c'è niente da vedere in un fotogramma solo, perché quello che decide è
l'accumulo. In {numref}`fig-formiche-feromone` ci sono sei giri della colonia
sul ponte doppio, con le due strade che cambiano spessore man mano che il
feromone si deposita.

```{figure} ../figures/formiche-feromone.svg
:name: fig-formiche-feromone
:alt: "Il formicaio a sinistra e il cibo a destra, uniti da due strade: una lunga che sale in alto e una corta che passa in basso. Giro dopo giro entrambe si ispessiscono, perché su entrambe passano formiche, ma la corta molto più in fretta, e il divario fra le due cresce. Il contatore sotto dice quante formiche su cento scelgono la corta: si parte da cinquanta e si arriva a ottantadue."
:width: 88%

Sei giri della colonia sul ponte doppio: lo spessore di ciascuna strada è il
feromone che ci si è accumulato sopra. Si parte in parità, cinquanta e
cinquanta, e si finisce con l'ottantadue per cento delle formiche sulla strada
corta. I numeri li calcola la figura, con le due formule qui sopra e niente
altro.
```

Due cose vale la pena guardare, e la seconda è quella che conta.

La prima è *dove* cambia la pendenza: il salto grosso è fra il primo e il
secondo giro (sedici punti, contro i sette del secondo e i quattro del terzo),
cioè quando le due strade sono ancora percorse dallo stesso numero di
formiche e a fare la differenza è **solo** il $Q/L$. Da lì in poi il vantaggio
si rinforza da sé.

La seconda è che **anche la strada lunga si ispessisce**. Non è un dettaglio del
disegno: è il difetto del meccanismo. Finché le formiche passano, il feromone si
accumula dappertutto e non se ne va più; quello che cresce è il *divario*, non
la differenza fra una traccia e nessuna traccia. Un sistema fatto così sa
premiare, ma non sa dimenticare, ed è esattamente il buco che la sezione
seguente va a tappare.

## L'evaporazione è l'esplorazione

Fin qui il meccanismo ha un difetto grosso, e vale la pena vederlo prima della
cura, perché è lo stesso difetto di molti sistemi che si auto-rinforzano. Il
feromone attira formiche, le formiche depositano feromone: il ciclo è
**positivo**, e un ciclo positivo lasciato a sé stesso non converge, esplode.
La prima strada trovata per caso diventa la più battuta, la più battuta diventa
l'unica, e la colonia si fossilizza su una soluzione che non ha nessun motivo di
essere buona: nel gergo dell'articolo è il **comportamento di stagnazione**, la
situazione in cui tutte le formiche fanno lo stesso giro e nessuna cerca più
niente.

La cura è una sola parola: il feromone **evapora**.

`````{tab} Elementare

Immagina che ogni sera metà del feromone se ne vada da solo. Una strada che
continua a essere usata non se ne accorge, perché ogni giorno riceve un
deposito nuovo. Una strada che è stata la migliore per un po' e poi non lo è
più si sbiadisce in fretta: partendo da cento, in cinque sere passa a cinquanta,
venticinque, dodici e mezzo, poco più di sei, poco più di tre. Dopo una
settimana è sotto l'uno per cento, come se non fosse mai esistita, e le formiche
tornano a provarci altrove.

Detta così sembra una perdita, ed è invece la cosa più preziosa dell'algoritmo.
Senza evaporazione la memoria del gruppo è definitiva: la prima strada scoperta
per caso resta la più marcata per sempre, e nessuna formica avrà mai occasione
di scoprire quella dietro l'angolo. Con l'evaporazione la memoria è un ricordo
che va tenuto vivo per restare. Il gruppo dimentica, e dimenticando continua a
guardarsi attorno.

`````

`````{tab} Superiore

L'aggiornamento della traccia, nella convenzione oggi standard, è

$$
\tau_{ij} \;\leftarrow\; (1-\rho)\,\tau_{ij} \;+\; \Delta\tau_{ij},
\qquad \rho \in (0, 1],
$$

dove $\rho$ è il **tasso di evaporazione**: la frazione di feromone che sparisce
a ogni ciclo, indipendentemente da quello che le formiche stanno facendo. (Due
avvertenze di notazione. Nell'articolo del 1996 la stessa lettera indica la
*persistenza*, e l'aggiornamento compare come
$\tau \leftarrow \rho\,\tau + \Delta\tau$, con $1-\rho$ a fare l'evaporazione:
è la stessa legge scritta dall'altro verso, e la letteratura successiva ha
invertito la convenzione. Inoltre $\rho$ è la terza cosa che questa lettera
indica nel capitolo, dopo la densità dello stormo e la correlazione fra
votanti: sono notazioni consolidate nei rispettivi campi, e qui vale quella
locale.)

Srotolando la ricorsione si vede che cosa sia davvero la traccia:

$$
\tau_{ij}(t) \;=\; \sum_{s \le t} (1-\rho)^{\,t-s}\;\Delta\tau_{ij}(s),
$$

cioè una **media mobile esponenziale** della qualità recente di quell'arco, con
i contributi vecchi pesati sempre meno. Due conseguenze quantitative. La prima:
la traccia non diverge. Se un arco riceve un deposito costante $\Delta$ a ogni
ciclo, la serie geometrica converge al punto fisso $\tau^{\ast} = \Delta/\rho$,
che con $\rho = 0{,}5$ vale il doppio di un singolo deposito. La seconda: la
somma dei pesi è $1/\rho$, quindi **$1/\rho$ è l'orizzonte di memoria** in
cicli. Con $\rho$ vicino a zero la colonia ricorda tutto e si fossilizza sul
primo ottimo locale; con $\rho$ vicino a uno dimentica a ogni giro e le formiche
tornano a essere golose e scorrelate. Il valore che gli autori trovano migliore
per questa variante sta esattamente in mezzo, $\rho = 0{,}5$ (l'unico numero che
vale lo stesso nelle due convenzioni, appunto perché sta in mezzo), cioè un
orizzonte di due cicli; e la ragione che ne danno è la migliore descrizione in
una riga del compromesso esplorazione-sfruttamento: l'algoritmo ha bisogno di
poter **dimenticare parte dell'esperienza passata** per sfruttare l'informazione
globale che sta arrivando adesso.

`````

## La memoria non sta negli individui

Adesso il punto che rende questa sezione parte di questo capitolo e non del
capitolo sull'ottimizzazione. Le formiche artificiali non si scambiano un solo
messaggio. Non si conoscono, non si nominano, non sanno nemmeno in quante
sono. Tutto quello che una formica sa delle altre lo legge **per terra**: la
loro esperienza è diventata una proprietà fisica dell'ambiente, e la traccia
sopravvive alle singole formiche che l'hanno lasciata.

Gli autori lo dicono in una frase che potrebbe stare in un manuale di sistemi
distribuiti: nell'Ant System un insieme di formiche comunica **modificando una
struttura dati globale**. Chi ha letto la sezione sulle topologie ha già
riconosciuto la forma e ha già il nome: è la **lavagna condivisa** di
Hearsay-II, ed è la **stigmergia** che lì abbiamo definito, il coordinamento
attraverso le tracce lasciate in uno spazio comune invece che attraverso
messaggi diretti. Le tre proprietà tornano tutte: il disaccoppiamento è massimo
(una formica in più non richiede di modificare nessun'altra), il centro non
decide ma conserva, e la provenienza si perde, perché il feromone su un arco è
un numero e non dice più chi ce l'ha messo.

La conseguenza fino a oggi è meno metaforica di quanto sembri. Una squadra di
agenti che si coordina lasciando file in una cartella condivisa, o note in un
documento che tutti possono leggere e riscrivere, sta facendo esattamente
questo: non si scrivono messaggi, si modifica uno stato comune e si reagisce a
come lo si trova. Cambia la taglia (il feromone è un numero, una nota è un
paragrafo) ma il regime di progetto è lo stesso, e con esso i problemi: la
contesa in scrittura, la provenienza da registrare a mano, e **che cosa fa
dimenticare** allo stato condiviso ciò che non serve più. Le formiche ce
l'hanno per costruzione; una cartella di file cresce e basta.

## Lo sciame di particelle

Il secondo classico del filone nasce da tutt'altra parte, e la sua origine
riporta dritti allo stormo dell'apertura. Nel 1995, alla International
Conference on Neural Networks di Perth, James Kennedy e Russell Eberhart
presentano la **particle swarm optimization** {cite}`kennedy1995particle`.
Una riga delle loro conclusioni spiega mezza storia: gli autori sono uno
psicologo sociale e un ingegnere elettronico. Erano partiti provando a simulare
uno stormo, ispirandosi ai *boids* di Reynolds {cite}`reynolds1987flocks` e ai
modelli di volo coordinato di Heppner e Grenander, e hanno scoperto che quel
giocattolo, tolti i pezzi giusti, **ottimizzava**.

Il racconto delle amputazioni è la parte istruttiva. Via la «pazzia», cioè il
rumore aggiunto a mano per rendere il volo credibile: non serviva. Via
l'allineamento con il vicino più prossimo (ogni agente copiava la velocità del
compagno più vicino, che è l'allineamento dei boids ridotto a un solo vicino):
senza, riportano gli autori, l'ottimizzazione va perfino un po' più in fretta, e
quello che resta non è più uno stormo, è uno sciame. Restano due sole
attrazioni: verso il punto migliore che quella particella ha trovato finora, e
verso il punto migliore che ha trovato il gruppo.

`````{tab} Elementare

Un gruppo di persone cerca il punto più basso di una valle nella nebbia. Ognuno
vede solo dove mette i piedi, e può misurare la quota lì dove si trova. Nessuno
ha la mappa.

Ciascuno si ricorda una cosa sola: il punto più basso in cui *lui* è passato. E
ne sente una sola: il punto più basso in cui è passato *qualcuno*, gridato a
tutti. A ogni passo tira un po' verso il proprio ricordo, un po' verso quello
del gruppo, e un po' tira dritto per dove stava già andando, perché ha una sua
velocità e non si ferma di colpo.

Quest'ultima cosa sembra un dettaglio ed è quella che fa funzionare tutto. Se
uno andasse solo dove è tirato, arriverebbe al punto migliore conosciuto e si
fermerebbe lì, insieme a tutti gli altri; ma siccome arriva lanciato, lo
supera, va a guardare un po' più in là, e ogni tanto scopre che più in là si
scende ancora. Gli autori hanno provato a togliere questa inerzia e il metodo ha
smesso di trovare i minimi buoni: le soluzioni migliori non stanno dove il
gruppo sta già puntando, stanno appena oltre.

`````

`````{tab} Superiore

Ogni particella $i$ è una coppia posizione-velocità
$(\mathbf{x}_i, \mathbf{v}_i)$ nello spazio delle soluzioni, e sono vettori: il
grassetto qui non è decorazione, perché tutto ciò che segue si applica
componente per componente. L'aggiornamento è di due righe, ripetute:

$$
\mathbf{v}_i \;\leftarrow\; w\,\mathbf{v}_i
\;+\; c_1\, \mathbf{r}_1 \odot (\mathbf{p}_i - \mathbf{x}_i)
\;+\; c_2\, \mathbf{r}_2 \odot (\mathbf{g} - \mathbf{x}_i),
\qquad
\mathbf{x}_i \;\leftarrow\; \mathbf{x}_i + \mathbf{v}_i,
$$

dove $\mathbf{p}_i$ è la posizione migliore visitata dalla particella $i$ (il
termine *cognitivo*), $\mathbf{g}$ la migliore visitata dall'intero sciame (il
termine *sociale*), $\mathbf{r}_1$ e $\mathbf{r}_2$ sono vettori di numeri
casuali uniformi in $[0,1]$ estratti daccapo a ogni passo, $\odot$ è il prodotto
componente per componente, mentre $c_1$, $c_2$ e l'**inerzia** $w$ sono scalari
che dosano le tre spinte. I tre addendi sono, nell'ordine, dove stavo andando,
dove sono stato meglio io, dove è stato meglio il gruppo.

Due precisazioni storiche. Nella formulazione del 1995 il peso $w$ non c'è: la
velocità precedente entra con coefficiente unitario, e l'inerzia come parametro
regolabile la introducono Shi ed Eberhart nel 1998, perché $w$ grande favorisce
l'esplorazione e $w$ piccolo la convergenza. E il valore originale
$c_1 = c_2 = 2$ non è arbitrario: moltiplicando per $2$ un numero uniforme in
$[0,1]$ si ottiene un fattore di media $1$, così che ciascuna delle due spinte,
**in media**, porti la particella esattamente sul proprio attrattore, e quindi
la faccia sorpassare circa una volta su due. Il sorpasso è deliberato: gli
autori riportano che rimuovendo il termine di inerzia (cioè sostituendo la
velocità invece di correggerla) l'algoritmo diventa inefficace nel trovare gli
ottimi globali. È la stessa ragione dell'evaporazione delle formiche, in veste
meccanica: un sistema che va solo dove è già andato bene smette di cercare.

`````

## Rimescolare invece di muoversi: gli algoritmi genetici

Formiche e particelle si **spostano**: c'è uno spazio, e ogni individuo ha una
posizione che aggiorna. La terza famiglia di questa cassetta degli attrezzi
rinuncia anche a quello, e cambia il verbo. Gli individui non si muovono: si
**riproducono**.

`````{tab} Elementare

Immagina di dover riempire uno zaino scegliendo fra venti oggetti, ognuno con
un peso e un valore, senza superare il limite di carico. Non c'è nessuna
pendenza da seguire: le soluzioni non sono punti su una collina, sono elenchi
di sì e no, e non esiste un «poco più a destra».

Un algoritmo genetico parte da una popolazione di zaini riempiti a caso, quasi
tutti mediocri, e ripete tre gesti che vengono dalla biologia.

**Selezione.** Chi vale di più ha più probabilità di fare figli. Il modo più
semplice è il torneo: si pescano due individui a caso e passa il migliore.

**Incrocio.** Da due genitori si fa un figlio prendendo la prima metà
dell'elenco dall'uno e la seconda dall'altro. È il gesto che le formiche e le
particelle non hanno, ed è quello che dà il nome alla famiglia.

**Mutazione.** Ogni tanto, a caso, si ribalta una scelta: un oggetto che c'era
esce, uno che non c'era entra. Serve a non restare prigionieri del materiale
genetico di partenza, ed è la stessa funzione dell'evaporazione nelle formiche
e dell'inerzia nelle particelle.

L'incrocio nasconde una scommessa, e conviene dirla, perché è il punto in cui
questi algoritmi funzionano o falliscono: **si sta assumendo che una buona
soluzione sia fatta di buoni pezzi**, e che i pezzi di due soluzioni decenti,
mescolati, possano darne una migliore. Sullo zaino l'assunzione regge (un buon
sottoinsieme di oggetti resta buono accanto a un altro). Su un problema dove
il valore dipende da tutte le scelte insieme, e spezzare l'elenco a metà
distrugge il senso di entrambe le metà, l'incrocio è solo rumore costoso.

`````

`````{tab} Superiore

Il quadro lo fissa John Holland {cite}`holland1975adaptation` nel 1975.
Una soluzione candidata è codificata come una stringa (il *genotipo*, nel caso
più semplice binaria) e la funzione obiettivo diventa la *fitness*. A ogni
generazione si applicano tre operatori:

- **selezione**, che campiona i genitori con probabilità crescente nella
  fitness (proporzionale alla fitness, cioè la «roulette», oppure per torneo,
  che è più robusto perché dipende solo dall'*ordine* e non dalla scala dei
  valori);
- **crossover**, che ricombina due genotipi (a un punto, a due punti,
  uniforme);
- **mutazione**, che perturba ogni gene con probabilità piccola.

Si aggiunge quasi sempre l'**elitismo**, cioè il trasferimento diretto del
migliore alla generazione successiva, senza il quale la ricerca può peggiorare
da una generazione all'altra.

La giustificazione classica dell'incrocio è l'ipotesi dei *building block*:
schemi parziali corti e buoni verrebbero propagati e combinati. È
un'argomentazione euristica più che un teorema, e il suo limite ha un nome
preciso, **epistasi**: quando il contributo di un gene dipende fortemente dagli
altri, spezzare il genotipo distrugge proprio l'informazione che si voleva
trasmettere, e il crossover degrada a mutazione macroscopica. La codifica non
è quindi un dettaglio implementativo: **è il progetto dell'algoritmo**, perché
decide quali pezzi sono separabili.

Rispetto alle altre due famiglie della sezione, la differenza operativa è che
lo spazio non deve avere una metrica. PSO ha bisogno di sommare posizioni e
velocità, quindi di uno spazio vettoriale; un algoritmo genetico ha bisogno
solo di saper ricombinare e perturbare, e questo lo rende applicabile a
permutazioni, alberi, grafi e programmi. Il caso in cui l'individuo è un
programma si chiama *programmazione genetica*.

Un secondo vantaggio distintivo, che né il gradiente né le altre metaeuristiche
danno gratis, è l'ottimizzazione **multi-obiettivo**. Poiché la popolazione è
un insieme e non un punto, la si può far convergere non su un ottimo ma
sull'intero **fronte di Pareto** dei compromessi fra obiettivi in conflitto
(accuratezza contro latenza, prestazione contro consumo): è ciò che fa NSGA-II
{cite}`deb2002fast`, ordinando la popolazione per dominanza invece che per un
punteggio scalare. Con un metodo a singolo punto bisognerebbe fissare i pesi
degli obiettivi in anticipo e rilanciare la ricerca per ogni compromesso.

`````

Lo zaino non è un esempio scelto a caso: è il tipo di problema su cui la
discesa del gradiente non ha proprio dove appoggiarsi. Nel codice che segue
l'istanza è fissata, e poiché ha solo venti oggetti possiamo permetterci il
lusso di conoscere la risposta vera, enumerando tutte le combinazioni: così
l'algoritmo si può giudicare invece che ammirare.

```python
import numpy as np
from itertools import product

# --- l'istanza: 20 oggetti, uno zaino che regge 60 kg (fissata una volta) ---
istanza = np.random.default_rng(7)
N, CAPIENZA = 20, 60
peso   = istanza.integers(4, 20, N)
valore = istanza.integers(5, 40, N)

def bonta(pop):                      # quanto vale uno zaino; 0 se sfonda il limite
    return np.where(pop @ peso <= CAPIENZA, pop @ valore, 0)

def genetico(seme, POP=60, GEN=80, P_MUT=0.03):
    rng = np.random.default_rng(seme)
    pop = rng.integers(0, 2, size=(POP, N))          # una popolazione di zaini a caso
    for _ in range(GEN):
        f = bonta(pop)
        elite = pop[f.argmax()].copy()               # il migliore non si perde mai
        s = rng.integers(0, POP, size=(POP, 2))      # selezione: torneo a due
        genitori = np.where((f[s[:, 0]] >= f[s[:, 1]])[:, None], pop[s[:, 0]], pop[s[:, 1]])
        taglio = rng.integers(1, N, size=(POP, 1))   # incrocio a un punto:
        maschera = np.arange(N)[None, :] < taglio    # meta' da un genitore, meta' dall'altro
        figli = np.where(maschera, genitori, genitori[rng.permutation(POP)])
        figli ^= (rng.random((POP, N)) < P_MUT)      # mutazione: qualche bit ribaltato
        figli[0] = elite
        pop = figli
    return int(bonta(pop).max())

esiti = [genetico(s) for s in range(10)]
ottimo = max(sum(v for v, b in zip(valore, c) if b)
             for c in product([0, 1], repeat=N)
             if sum(p for p, b in zip(peso, c) if b) <= CAPIENZA)

print("dieci esecuzioni:", esiti)
print("ottimo vero (forza bruta su 2^20 = 1 048 576 combinazioni):", ottimo)
print(f"quante volte lo trova: {esiti.count(ottimo)}/10, con 4800 zaini provati su un milione")
```

```text
dieci esecuzioni: [228, 228, 228, 228, 224, 228, 228, 228, 228, 224]
ottimo vero (forza bruta su 2^20 = 1 048 576 combinazioni): 228
quante volte lo trova: 8/10, con 4800 zaini provati su un milione
```

Il risultato dice due cose insieme, e vanno tenute insieme. La prima è che
provando meno di mezzo per cento delle combinazioni si arriva otto volte su
dieci all'ottimo esatto, e le altre due volte al $98\%$ di esso: per un
problema in cui non esiste alcuna pendenza da seguire, è molto. La seconda è
che quel «otto volte su dieci» non si può eliminare. Un algoritmo genetico non
dà garanzie, e soprattutto **non dice quanto gli è mancato**: qui lo sappiamo
solo perché venti oggetti si possono enumerare a mano. Con quaranta oggetti il
confronto non esisterebbe, e la risposta trovata avrebbe esattamente lo stesso
aspetto.

Nel machine learning questa famiglia compare in due punti. Il primo è la
**ricerca di architetture**, e conviene distinguere subito le due strade perché
si confondono spesso. La rete base di EfficientNet, ricordata nel capitolo sul
deep learning, viene da una ricerca automatica multi-obiettivo guidata dal
**reinforcement learning**, non dall'evoluzione. L'evoluzione è l'altra strada
principale, e il suo esemplare è AmoebaNet {cite}`real2019regularized`, dove le
architetture **mutano** e le migliori sopravvivono, con una selezione a torneo
che scarta anche le più vecchie. Vale la pena notare che quell'algoritmo il
crossover non ce l'ha, ed è coerente con la scommessa dichiarata poche righe fa:
in un'architettura i pezzi non sono separabili, perché un blocco che funziona
bene in una rete può essere pessimo in un'altra, quindi la scommessa non
reggerebbe e l'algoritmo si limita a non farla. Il secondo punto è ovunque
l'obiettivo non sia derivabile: scegliere iperparametri discreti, potare una
rete decidendo *quali* pezzi togliere, ottimizzare una pipeline di
preelaborazione.

## Perché non usare il gradiente

Sia le formiche sia le particelle hanno una proprietà che va guardata in faccia:
non usano mai la **derivata** della funzione da minimizzare. Vedono solo il suo
valore, in un punto alla volta. Il capitolo di matematica ha dedicato una
sezione alla discesa del gradiente, che è il metodo con cui si addestra ogni
rete di questo libro; qui abbiamo un'altra famiglia, e il confronto va fatto
per bene, perché è il punto in cui la divulgazione su questi metodi diventa
disonesta.

`````{tab} Elementare

La differenza è quella fra sentire la pendenza sotto i piedi e doverla scoprire
provando. Chi sente la pendenza sa subito da che parte si scende, e fa un passo
nella direzione giusta. Chi non la sente deve fare un passo a caso, misurare la
quota dove è arrivato, e capire dopo se era la direzione giusta.

Su una collina liscia il primo arriva in fondo in una manciata di passi e il
secondo in moltissimi: non c'è partita. Il secondo vince in due situazioni,
però. La prima è un terreno pieno di buche: chi segue la pendenza finisce nella
buca più vicina e lì resta, convinto di essere in fondo, mentre di un gruppo
sparso per la valle è probabile che qualcuno sia partito vicino a quella giusta.
La seconda è quando la pendenza non si può proprio sentire: se il «terreno» è
l'ordine in cui visitare venti città, o quale macchinario assegnare a quale
lavorazione, non esiste nessuna direzione in cui muoversi di un millimetro, ed
esiste solo provare.

`````

`````{tab} Superiore

I metodi di questa sezione sono **senza derivate**: interrogano la funzione
obiettivo come una scatola nera e non ne richiedono né differenziabilità né
continuità. Il loro dominio proprio è dove il gradiente non c'è, non si calcola
o non informa: funzioni non differenziabili, spazi **combinatori** (il commesso
viaggiatore non ha un gradiente: ha permutazioni), valutazioni **rumorose** o
prodotte da una simulazione, e paesaggi molto multimodali dove il gradiente è
informativo solo dentro il bacino in cui si nasce.

Il prezzo si paga in **valutazioni della funzione obiettivo**, e in alta
dimensione diventa proibitivo, per una ragione precisa. Con la
retropropagazione una sola passata all'indietro produce tutte le $d$ derivate
parziali di $\mathcal{L}$ rispetto ai parametri $\theta$, a un costo
dell'ordine di una passata in avanti: l'informazione per passo cresce con $d$
mentre il costo no. Un metodo senza derivate deve invece **stimare** una
direzione utile a partire da valori scalari, e le valutazioni necessarie
crescono almeno linearmente con $d$. È il motivo per cui nessuno addestra con
uno sciame una rete da centinaia di milioni di parametri, e insieme il motivo
per cui gli sciami restano vivi dove $d$ è piccolo e ogni valutazione è cara
(taratura di iperparametri, progettazione ingegneristica, instradamento,
schedulazione). Va aggiunto, per onestà, che di questi metodi **non esiste una
garanzia di convergenza all'ottimo globale** in tempo utile: sono euristiche,
funzionano bene su molte istanze e nessuno può promettere che funzionino sulla
prossima. Chi li presenta come alternativa generale alla discesa del gradiente
sta vendendo qualcosa.

`````

## Uno sciame in venti righe

Il modo più rapido di crederci è farlo girare. La funzione di prova è la
**Rastrigin** in due dimensioni, e conviene immaginarsela così: una conca
larghissima e regolare, che scende dolcemente verso il centro, sulla quale
qualcuno ha passato una grattugia, cioè un'ondulazione fitta e ordinata che
scava una fossetta attorno a ogni coppia di numeri interi. Il fondo vero è al
centro e vale zero; le fossette sono centinaia, e dal fondo di ognuna tutte le
direzioni salgono. È il paesaggio fatto apposta per mettere in crisi chi segue
la pendenza.

```python
import numpy as np


# Rastrigin in due dimensioni: minimo globale in (0, 0), dove vale 0.
# Tutt'attorno un reticolo di conche locali, attorno a ogni coppia di interi.
def rastrigin(X):
    return 10 * X.shape[1] + np.sum(X**2 - 10 * np.cos(2 * np.pi * X), axis=1)


rng = np.random.default_rng(7)     # seme fisso: il risultato e' riproducibile
n, d = 30, 2                       # trenta particelle in due dimensioni
w, c1, c2 = 0.73, 1.50, 1.50       # inerzia, spinta personale, spinta sociale

X = rng.uniform(-5.12, 5.12, (n, d))       # posizioni iniziali, sparse a caso
V = rng.uniform(-1.0, 1.0, (n, d))         # velocita' iniziali
P, fP = X.copy(), rastrigin(X)             # miglior punto di ogni particella
g = int(np.argmin(fP))                     # indice del migliore del gruppo

for t in range(1, 61):
    r1, r2 = rng.random((n, d)), rng.random((n, d))
    V = w * V + c1 * r1 * (P - X) + c2 * r2 * (P[g] - X)
    X = np.clip(X + V, -5.12, 5.12)        # nessuno esce dal dominio
    f = rastrigin(X)
    meglio = f < fP                        # chi ha battuto il proprio record
    P[meglio], fP[meglio] = X[meglio], f[meglio]
    g = int(np.argmin(fP))
    if t % 10 == 0:
        print(f"iterazione {t:3d}   f = {fP[g]:.6f}   "
              f"x = ({P[g][0]:+.4f}, {P[g][1]:+.4f})")
```

```text
iterazione  10   f = 1.948778   x = (-0.9332, +0.0325)
iterazione  20   f = 0.000262   x = (+0.0008, -0.0008)
iterazione  30   f = 0.000262   x = (+0.0008, -0.0008)
iterazione  40   f = 0.000262   x = (+0.0008, -0.0008)
iterazione  50   f = 0.000213   x = (+0.0010, -0.0004)
iterazione  60   f = 0.000168   x = (+0.0009, +0.0000)
```

La riga da guardare è la prima. Alla decima iterazione il punto migliore che lo
sciame conosce è $(-0{,}93,\ 0{,}03)$, che non è il minimo globale: è dentro la
conca locale accanto, sul cui fondo la funzione vale circa $1$ e nel punto
trovato $1{,}95$. Un metodo a gradiente partito da lì scivolerebbe in fondo a
quella conca e ci resterebbe per sempre, perché dal fondo tutte le direzioni
salgono. Lo sciame ne esce entro la ventesima, e ne esce senza aver capito
niente: una particella era semplicemente arrivata più in là del punto migliore
conosciuto, e più in là si scendeva. Dalla ventesima in poi il gruppo raffina un
valore già a tre zeri dopo la virgola.

Un paio di conti per non farsi un'idea sbagliata. Le valutazioni della funzione
sono $30 \times 61 = 1830$: in due dimensioni sono niente, in mille sarebbero
ancora $1830$ e non basterebbero. E il risultato è **probabilistico**: ripetendo
lo stesso esperimento con trecento semi diversi, e contando come riuscite le
prove che chiudono sotto $10^{-2}$, lo sciame arriva al minimo globale in $277$
casi su $300$, cioè poco più di nove volte su dieci, non sempre.

Resta il confronto con il gradiente, ed è il punto in cui la divulgazione su
questi metodi imbroglia quasi sempre. Fatta partire da **un solo** punto preso a
caso, una discesa del gradiente ordinaria (passo $0{,}005$, duemila iterazioni)
chiude sotto $10^{-2}$ una volta su trecento, e nelle altre duecentonovantanove
si ferma ordinatamente nella fossetta in cui è nata. Duecentosettantasette contro
uno: un confronto splendido e scorretto, perché schiera trenta esploratori
contro uno solo, e viola la clausola che questo stesso capitolo ha enunciato
come regola vincolante due sezioni fa, **a parità di budget**.

Rifacciamolo per bene. Al gradiente si danno **trenta ripartenze** per prova,
cioè gli stessi trenta punti iniziali che ha lo sciame, e si tiene il migliore
dei trenta. Allora chiude $62$ prove su $300$, cioè una su cinque, contro le
nove su dieci dello sciame. Lo sciame vince ancora, e vince nettamente, ma vince
tre volte tanto e non trecento. Se poi si guarda la spesa, il confronto è
perfino generoso verso il gradiente: le trenta discese sono sessantamila passi,
contro le $1830$ valutazioni dello sciame.

Due avvertenze sui numeri, perché il primo è più fragile di quanto sembri. La
partenza singola non è riproducibile come il resto del capitolo: cambiando seme
si ottiene qualunque cosa fra $0$ e $4$ su $300$, e il risultato dipende dai due
iperparametri al punto che con un passo di $0{,}01$ scende a zero. Quell'«una
volta su trecento» è l'esito di un'esecuzione, non una costante, e conviene
affiancargli la ragione geometrica, che invece è solida: la fossetta centrale è
larga circa $1 \times 1$ su un dominio di lato $10{,}24$, cioè occupa poco meno
dell'$1\%$ dell'area, e un punto preso a caso ci cade circa una volta su cento.
La seconda avvertenza è che a trenta ripartenze il conto elementare
$1 - (1 - 0{,}0095)^{30} \approx 0{,}25$ prevederebbe $75$ prove su $300$ invece
di $62$: la differenza sono le partenze nate nella fossetta giusta che non
arrivano abbastanza in fondo entro duemila passi. Nascere nel bacino buono è
necessario, non sufficiente.

Un'ultima nota sui tre numeri in cima al programma, quelli che pesano le tre
spinte (tirare dritto per dove stavo andando, tornare dove sono stato meglio io,
andare dove è stato meglio il gruppo). Non sono i valori del 1995 ma quelli oggi
standard, e non sono stati trovati provando: vengono da un conto che dice per
quali valori la velocità delle particelle **non esplode**. Con spinte troppo
forti lo sciame si sparpaglia e non torna più; con questi tre numeri sta insieme
da sé, e nessuno deve tarare a mano l'ampiezza dei passi.

## Venticinque agenti in un paese

Gli sciami mettono molte unità stupide a risolvere un problema. Con i modelli di
linguaggio si può fare una cosa che prima non si poteva: mettere molte unità
**non** stupide a fare qualcosa che un problema di ottimizzazione non è, cioè
comportarsi. Il lavoro di riferimento è quello di Park e colleghi del 2023
{cite}`park2023generative`, che il capitolo sugli Agenti ha già presentato nella
sezione sulla memoria: venticinque agenti in un paese simulato, ciascuno con un
archivio di ricordi in linguaggio naturale. Il risultato più citato è un
comportamento emerso: un'agente decide di dare una festa di San Valentino,
l'invito si propaga di bocca in bocca senza che nessuno lo instradi, e alla fine
tredici agenti su venticinque ne sanno qualcosa e cinque si presentano.

Non ripetiamo l'architettura, che è già stata descritta lì: flusso di
osservazioni, recupero, riflessione, pianificazione. Vale la pena guardare da
vicino il pezzo che regge tutto, e che è anche il più imitato senza capirlo: la
**funzione di recupero a tre termini**. Risponde alla domanda di qualunque
memoria grande: fra diecimila ricordi, quali sono i pochi che vanno messi nel
contesto **adesso**?

`````{tab} Elementare

Tre criteri, e nessuno dei tre basta da solo. Quanto è **recente** il ricordo,
quanto è **importante**, e quanto **c'entra** con quello che sto facendo. Chi
usa solo il primo si ricorda l'ultima cosa successa; chi usa solo il secondo si
ripete addosso sempre lo stesso trauma; chi usa solo il terzo pesca frasi che
somigliano alla domanda ma sono di sei mesi fa.

Prima di sommarli bisogna però saperli misurare, e la **recenza** si misura
così: ogni ora che passa il ricordo perde mezzo punto percentuale di freschezza,
sempre lo stesso mezzo punto sul valore che gli era rimasto. Da lì escono i tre
numeri della prima colonna qui sotto: dopo un'ora resta $0{,}995$, dopo
cinquanta ore $0{,}778$, dopo duecento ore $0{,}367$, cioè poco più di un terzo.

Il guaio è sommarli, perché sono misurati in unità diverse. L'importanza è un
voto da 1 a 10 che l'agente si dà da sé, gli altri due sono numeri fra zero e
uno. Facciamo il conto su tre ricordi in gara:

| ricordo | recenza | importanza | pertinenza | somma diretta |
|---|---|---|---|---|
| A: di un'ora fa, molto attinente | $0{,}995$ | $3$ | $0{,}82$ | $4{,}82$ |
| B: di duecento ore fa, drammatico | $0{,}367$ | $8$ | $0{,}44$ | $8{,}81$ |
| C: di cinquanta ore fa, così così | $0{,}778$ | $5$ | $0{,}60$ | $6{,}38$ |

Sommandoli così vince B, poi C, poi A: cioè esattamente l'ordine
dell'importanza, e gli altri due criteri non hanno contato niente. Ovvio: un
voto che va da 1 a 10 schiaccia due numeri che vanno da 0 a 1.

La cura è mettere le tre colonne sulla stessa scala prima di sommarle: in ogni
colonna il migliore prende 1, il peggiore prende 0, gli altri stanno in mezzo in
proporzione. Rifatti i conti, A totalizza $2{,}00$, C fa $1{,}48$ e B si ferma a
$1{,}00$: la classifica si è **rovesciata**, e il ricordo appena successo e
attinente batte quello drammatico e vecchio. Non è un dettaglio implementativo:
è la differenza fra un agente che ragiona su quello che sta succedendo e uno
ossessionato dal proprio passato più intenso.

`````

`````{tab} Superiore

Il punteggio di recupero è la somma di tre segnali,

$$
s(m, q) \;=\; \alpha_{\text{rec}}\,\widetilde{\text{rec}}(m)
\;+\; \alpha_{\text{imp}}\,\widetilde{\text{imp}}(m)
\;+\; \alpha_{\text{rel}}\,\widetilde{\text{rel}}(m, q),
$$

dove $m$ è un ricordo, $q$ la situazione corrente e la tilde indica che ogni
termine è stato riscalato con un **min-max** nell'intervallo $[0,1]$ prima della
somma. Nel lavoro originale i tre pesi valgono tutti $1$, il che rende la
normalizzazione l'unico meccanismo che impedisce al termine con l'escursione
più ampia di dominare: sommare direttamente un voto in $[1,10]$ e due grandezze
in $[0,1]$ equivale a ordinare per il solo voto. E poiché il riscalamento è
relativo all'insieme dei candidati, il punteggio **non è assoluto**: lo stesso
ricordo vale diversamente a seconda della compagnia.

I tre segnali. La **recenza** decade esponenzialmente nel tempo simulato
trascorso dall'ultimo recupero di quella memoria, con fattore $0{,}995$ per ora:
l'emivita è $\ln 0{,}5 / \ln 0{,}995 \approx 138$ ore, poco meno di sei giorni
simulati, e il tempo caratteristico è all'incirca $1/0{,}005 = 200$ ore (il
valore esatto, $-1/\ln 0{,}995$, è $199{,}5$). È la stessa forma
dell'evaporazione del feromone, con un $\rho$ molto più piccolo, e il paragone è
istruttivo: la colonia dimentica in due cicli perché deve continuare a
esplorare, un agente in duecento ore perché deve restare la stessa persona.
La **pertinenza** (*relevance*) è la similarità del coseno fra l'embedding del
ricordo e quello della query, cioè il recupero denso già visto nel RAG.
L'**importanza** è l'unica anomala: non si calcola, si **chiede al modello**,
che assegna alla memoria un voto di salienza da 1 a 10 nel momento in cui la
scrive, con tutti i pregiudizi che ha su che cosa conti in una vita.

Sopra i tre segnali sta un quarto meccanismo, la **riflessione**, e ciò che
merita attenzione è quando scatta: non a orario fisso, ma quando la somma delle
importanze degli eventi recenti supera una soglia (150 nella loro
implementazione, che nei loro esperimenti si traduce in due o tre riflessioni al
giorno). La cadenza è quindi guidata dagli eventi e non dall'orologio: una
giornata piatta non produce riflessioni, una densa ne produce diverse. L'agente
si pone allora le domande più salienti sul proprio periodo recente, risponde con
proposizioni astratte e
le **riscrive nel flusso** come ricordi nuovi, con la loro importanza e la loro
recenza. È una retroazione: le sintesi competono con le osservazioni grezze nel
recupero successivo, e sopra le prime riflessioni se ne formano altre. Ne esce
un albero di astrazioni costruito dal basso, ed è anche il punto delicato
dell'architettura, perché un'inferenza sbagliata in basso diventa una premessa
a tutti i livelli sopra.

`````

## Che cosa dimostra una simulazione di persone

Le trascrizioni di questi esperimenti sono convincenti. Gli agenti si invitano,
si ricordano di essersi conosciuti, si giustificano se arrivano tardi. È il
momento di essere precisi su che cosa questo autorizzi a concludere, perché la
tentazione di usare simulazioni del genere come evidenza sul comportamento umano
è forte e il salto non è consentito.

Il punto sta in una parola che gli autori usano con cura e che chi li cita
spesso lascia cadere: gli agenti producono comportamenti **credibili**
(*believable*), non veri. La credibilità è una proprietà del modello di
linguaggio da cui provengono, non un risultato dell'esperimento. Un modello
addestrato su enormi quantità di testo umano è, per costruzione, una macchina
per produrre continuazioni verosimili di testo umano; quando gli si chiede di
comportarsi come una persona, il fatto che il risultato somigli a una persona
**non è una scoperta**, è la specifica. Peggio: la nostra sensazione di aver
visto qualcosa di vero cresce proprio con la qualità del modello, cioè con la
sua abilità a produrre testo convincente, che è la variabile meno legata alla
verità di tutte. Le mani avanti se le mettono gli autori stessi, in una nota a
piè di pagina: i loro agenti, scrivono, puntano a dare un senso di credibilità
come i personaggi animati della Disney, e non implicano nessuna agentività vera.

Credibile non vuol dire predittivo, ed è la solita distinzione fra somigliare e
prevedere. Perché una simulazione dicesse qualcosa sulle società reali dovrebbe
riprodurre non i singoli comportamenti verosimili, ma le **distribuzioni** di
quei comportamenti: quante persone su venticinque davvero verrebbero alla festa,
e in quali condizioni nessuna. Su questo non c'è nessuna garanzia, e ce ne sono
anzi di contrarie: un modello di linguaggio riflette le proporzioni del proprio
corpus di addestramento, non quelle della popolazione che si vorrebbe studiare,
e tende a produrre risposte medie e consensuali dove una popolazione vera è
dispersa e conflittuale. Che tredici agenti su venticinque abbiano saputo della
festa è un fatto sulla simulazione, non una stima sulla diffusione di un invito
in un paese.

Ne discende una regola d'uso netta. Come **generatore di ipotesi** queste
simulazioni sono legittime e utili: fanno emergere dinamiche a cui non si era
pensato, permettono di provare a costo quasi nullo interfacce e scenari prima di
metterci delle persone, e sono un banco di prova per architetture di agenti (il
loro contributo principale). Come **prova** non valgono niente, e nessuna
quantità di trascrizioni convincenti le avvicina a una prova, perché ciò che le
rende convincenti è esattamente ciò che le rende inaffidabili. Chi presenta
l'esito di una simulazione come un risultato sulle persone fa con il testo
quello che nessuno accetterebbe con i numeri: chiamare dato ciò che è un'uscita
del proprio modello.

## La stessa manopola, girata su sistemi diversi

Il capitolo si chiude dove è cominciato. Abbiamo contato quanto costa
coordinarsi, disegnato le forme del grafo di comunicazione, tipizzato i messaggi
e le regole di decisione, visto che cosa succede quando gli agenti imparano
insieme, e siamo finiti su sistemi in cui i partecipanti non ragionano affatto.
Cambia tutto da una sezione all'altra: la taglia dei partecipanti, il loro
costo, perfino se si parlino o no. Non cambia la variabile di progetto, che è
sempre la regola di interazione. I sei o sette vicini topologici dello storno,
gli archi che il progettista concede o nega, la performativa scritta in cima al
messaggio, la ricompensa condivisa o individuale, il tasso di evaporazione del
feromone: sono la stessa manopola, girata su sistemi diversi.

È la tesi dell'apertura, arrivata in fondo intatta: il comportamento di un
gruppo è una proprietà del protocollo di interazione più che della bravura dei
singoli. Vale per gli storni sopra Termini, che con una regola metrica invece
che topologica si sfalderebbero nel momento peggiore
{cite}`ballerini2008interaction`; vale per una colonia di formiche artificiali,
che con la stessa formula e un parametro di evaporazione diverso o esplora per
sempre o si fossilizza sul primo tentativo. E vale per la squadra di agenti che
avete in mente di costruire: la domanda utile non è quale modello mettere dentro
ciascuno, ma che cosa può scrivere ciascuno, a chi, quando, e chi decide dopo.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Prima dei modelli di linguaggio il multi-agente era soprattutto
  **ottimizzazione**: tante unità quasi banali, nessuno che comanda, e una
  soluzione che emerge dall'interazione. Il capostipite è l'**ottimizzazione a
  colonia di formiche** {cite}`dorigo1996ant`, nata al Politecnico di Milano fra
  il 1991 e il 1992 (un articolo di convegno e la tesi di dottorato di Marco
  Dorigo). Fra due strade verso lo stesso cibo, quella corta si percorre più
  spesso e accumula più traccia: è il tempo a fare la misura, senza che nessuna
  formica confronti niente. E si lascia traccia in quantità proporzionale a
  quanto è buono il giro appena finito, così la traccia registra il merito e non
  il traffico.
- L'**evaporazione è l'esplorazione**. Se ogni sera metà della traccia se ne va
  da sola, una strada che continua a essere usata non se ne accorge e una strada
  abbandonata sparisce in una settimana; quanto lentamente evapora dice per
  quanti giri il gruppo ricorda. Senza evaporazione la prima strada trovata per
  caso resta la più marcata per sempre e la colonia si fossilizza.
- La memoria del gruppo non sta negli individui, sta nell'**ambiente**: è la
  **stigmergia**, cioè la lavagna condivisa della sezione sulle topologie. Una
  squadra di agenti che si coordina lasciando file in una cartella comune fa
  esattamente questo, con gli stessi problemi (chi scrive mentre un altro
  scrive, chi ha messo lì una certa cosa, e che cosa fa dimenticare allo stato
  comune ciò che non serve più).
- Nello **sciame di particelle** {cite}`kennedy1995particle`, nato togliendo
  pezzi a una simulazione di stormo {cite}`reynolds1987flocks`, ognuno cerca il
  punto più basso della valle nella nebbia tirando un po' verso il proprio
  ricordo, un po' verso il punto migliore che ha trovato il gruppo, e un po'
  dritto per dove stava già andando. È quest'ultima spinta a far superare il
  punto migliore conosciuto e a guardare appena più in là: senza, il metodo
  smette di trovare i minimi buoni.
- Gli **algoritmi genetici** {cite}`holland1975adaptation` cambiano verbo: gli
  individui non si spostano, si riproducono (selezione, incrocio, mutazione, più
  il migliore che passa sempre alla generazione dopo). L'incrocio è una
  scommessa dichiarata, che una buona soluzione sia fatta di buoni pezzi
  staccabili, e cade quando ogni scelta dipende troppo da tutte le altre: **come
  si scrive la soluzione è il progetto** dell'algoritmo. In cambio a questi
  algoritmi non serve nessuna pendenza da seguire: basta saper mescolare due
  soluzioni e cambiarne un pezzo a caso. Va bene quindi anche una soluzione che
  è un elenco di sì e no, come gli oggetti da mettere nello zaino, o un ordine,
  come la sequenza in cui visitare venti città. E siccome in gara non c'è un
  candidato solo ma una popolazione intera, sparsa, è più probabile che qualcuno
  sia partito vicino alla risposta giusta. Sullo zaino a venti oggetti trova la
  risposta esatta otto volte su dieci provando meno di mezzo per cento delle
  combinazioni, e non dice mai quanto gli è mancato.
- Questi metodi **non sentono la pendenza**: provano un punto e misurano la
  quota. Servono dove la pendenza non c'è, non si calcola o non informa
  (terreni pieni di buche, misure rumorose, scelte in cui non ci si può
  spostare di un millimetro, come l'ordine in cui visitare venti città), e si
  pagano in tentativi: sulla valle piena di fossette dell'esempio lo sciame
  trova il fondo vero in $277$ prove su $300$. Il confronto va però fatto
  **a parità di esploratori**, altrimenti si bara: una discesa del gradiente
  lanciata da un punto solo ci arriva una volta su trecento, ma lanciata dagli
  stessi trenta punti dello sciame ci arriva una volta su cinque. Lo sciame
  vince tre volte, non trecento. E quando le variabili sono tantissime il
  rapporto si rovescia: non sono un'alternativa generale.
- Nelle **società simulate** {cite}`park2023generative` il pezzo da capire è
  come si scelgono i ricordi da rimettere davanti all'agente: quanto è recente,
  quanto è importante, quanto c'entra con quello che sta facendo, e i tre
  criteri vanno messi **sulla stessa scala** prima di sommarli, altrimenti vince
  sempre quello con i numeri più grandi. Le riflessioni scattano per accumulo di
  cose importanti, non a orologio, e rientrano in memoria. Ma quegli agenti
  producono comportamenti **credibili** (*believable*), che è ciò che un modello
  di linguaggio sa fare per costruzione e non una scoperta sul comportamento
  umano: valgono come **generatore di ipotesi**, non come **prova**, perché sono
  convincenti proprio in quanto il modello è addestrato a convincere.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Prima degli LLM il multi-agente era soprattutto **ottimizzazione**: molte
  unità quasi banali, nessun controllore centrale, e una soluzione che emerge
  dall'interazione. Il capostipite è l'**ottimizzazione a colonia di formiche**
  {cite}`dorigo1996ant`, nata al Politecnico di Milano fra il 1991 e il 1992 (un
  articolo di convegno e la tesi di dottorato di Marco Dorigo). Una formica
  sceglie l'arco con probabilità
  $p_{ij} \propto \tau_{ij}^{\alpha}\eta_{ij}^{\beta}$ (feromone contro
  visibilità) e deposita $Q/L_k$, tanto più quanto è buono il giro che ha
  costruito: la traccia registra il merito, non il traffico.
- L'**evaporazione è l'esplorazione**. Con
  $\tau_{ij} \leftarrow (1-\rho)\tau_{ij} + \Delta\tau_{ij}$ la traccia è una
  media mobile esponenziale con orizzonte $1/\rho$ cicli; senza evaporazione il
  rinforzo positivo fossilizza la colonia sul primo cammino trovato per caso
  (comportamento di stagnazione).
- La memoria del gruppo non sta negli individui, sta nell'**ambiente**: è la
  **stigmergia**, cioè la lavagna condivisa della sezione sulle topologie. Una
  squadra di agenti che si coordina lasciando file in una cartella condivisa fa
  esattamente questo, con gli stessi problemi (contesa, provenienza, e che cosa
  fa dimenticare allo stato comune ciò che non serve più).
- Nella **particle swarm optimization** {cite}`kennedy1995particle`, nata
  togliendo pezzi a una simulazione di stormo alla Reynolds
  {cite}`reynolds1987flocks`, ogni particella combina inerzia, attrazione verso
  il proprio miglior punto e verso il miglior punto del gruppo. Il sorpasso è
  voluto: senza inerzia il metodo smette di trovare gli ottimi buoni.
- Gli **algoritmi genetici** {cite}`holland1975adaptation` cambiano verbo: gli
  individui non si spostano, si ricombinano (selezione, incrocio, mutazione,
  più l'elitismo). L'incrocio è una scommessa esplicita, che una buona
  soluzione sia fatta di buoni pezzi separabili, e cade quando i geni
  interagiscono troppo (*epistasi*): la **codifica è il progetto**
  dell'algoritmo. In cambio non serve una metrica sullo spazio, quindi si
  applica a permutazioni, alberi e programmi, e la popolazione permette di
  inseguire un intero **fronte di Pareto** invece di un punto solo
  {cite}`deb2002fast`. Sullo zaino a venti oggetti trova l'ottimo esatto otto
  volte su dieci provando meno di mezzo per cento delle combinazioni, e non
  dice mai quanto gli è mancato.
- Questi metodi **non usano il gradiente**, quindi servono dove il gradiente non
  esiste, non si calcola o non informa (funzioni non differenziabili,
  valutazioni rumorose, spazi combinatori), e pagano in valutazioni della
  funzione obiettivo: sulla Rastrigin in due dimensioni lo sciame trova il
  minimo globale in 277 prove su 300 con 1830 valutazioni. Il termine di
  paragone va preso **a parità di budget**, come impone la regola prudente del
  «Costo del coordinamento»: la discesa del gradiente a partenza singola chiude
  1 prova su 300, ma con trenta ripartenze, cioè con gli stessi trenta punti
  iniziali dello sciame, ne chiude 62. In alta dimensione il rapporto si
  rovescia, e non sono un'alternativa generale.
- Nelle **società simulate** {cite}`park2023generative` il pezzo da capire è il
  recupero a tre termini (recenza, importanza, pertinenza) **normalizzati** e
  sommati con pesi uguali: senza normalizzazione vince sempre il termine con
  l'escursione più ampia. Le riflessioni scattano per accumulo di importanza,
  non a orologio, e rientrano in memoria. Ma gli agenti producono comportamenti
  **credibili** (*believable*), che è una proprietà del modello di linguaggio e
  non una scoperta sul comportamento umano: legittime come **generatore di
  ipotesi**, prive di valore come **prova**, perché sono convincenti proprio in
  quanto il modello è addestrato a convincere.
```

`````
