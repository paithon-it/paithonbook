# Imparare dal suono senza etichette

Chi trascrive migliaia di ore di audio? Qualcuno deve farlo: per insegnare a
una macchina a riconoscere il parlato servono coppie (audio, testo), e quel
testo lo scrive una persona, ascoltando e battendo a tastiera. È lento, costa,
e per molte lingue del mondo semplicemente non esiste. Eppure di audio *non*
etichettato ce n'è a valanga: podcast, audiolibri, video, archivi radiofonici,
registrazioni di ogni tipo; montagne di suono che nessuno ha mai trascritto.

C'è uno spreco evidente in tutto questo, e una domanda che se ne ricava: e se
la macchina imparasse la struttura del suono *da sola*, ascoltando quelle
montagne di audio grezzo, e usassimo poi ciò che ha imparato per i compiti
veri (riconoscere il parlato, classificare suoni, generare musica) con *poche*
etichette? È l'idea dell'apprendimento **auto-supervisionato**
(*self-supervised*).

C'è una parola che regge tutto il resto della sezione, e conviene fissarla
subito: **rappresentazione**. È il gruppetto di numeri con cui un modello si
tiene in mente un pezzetto di suono: qualche centinaio di numeri ogni venti
millesimi di secondo, cioè cinquanta gruppetti per ogni secondo di audio, che
nessuno ha scritto a mano e che nessuno saprebbe leggere uno per uno.

Quando è che una rappresentazione è *migliore* di un'altra? Immagina di
appendere ogni pezzetto di suono in un punto di una stanza enorme, scegliendo il
punto in base ai suoi numeri. Una buona rappresentazione è quella che appende
vicini i pezzetti che si somigliano davvero e lontani quelli diversi: tutte le
«s» in un angolo, tutte le «a» in un altro, i colpi di tamburo da tutt'altra
parte. Se la stanza è ordinata così, qualunque domanda arrivi dopo («che vocale
è?», «che strumento sta suonando?») si risponde tracciando una riga fra due
zone, e diventa facile. Anche lo spettrogramma della prima sezione era una
rappresentazione, ma l'avevamo disegnata noi; qui il modello se la costruisce da
sé.

Nel testo scritto lo stesso passaggio è già avvenuto, ed è il precedente da cui
questa storia nasce. Prima ogni parola aveva il suo gruppetto di numeri
**fisso**, identico in qualunque frase (sono i *word embedding*, word2vec e
simili, visti in [Rappresentare il
testo](../NaturalLanguageProcessing/rappresentare-testo.md)); poi si è passati a numeri che
**cambiano con la frase intorno**, così che la «pesca» del contadino e la
«pesca» del pescatore smettano di essere la stessa cosa, ed è il salto dei
grandi modelli pre-addestrati come BERT. Nell'audio è successo negli ultimi
anni, e questa sezione racconta come.

Diamo per acquisite le rappresentazioni di base del suono (campionamento,
spettrogramma, scala mel, MFCC) che abbiamo costruito nella sezione *Dal suono
alle feature*: qui non ripartiamo dall'onda di pressione, ma da come un
modello impara *da sé* una rappresentazione migliore.

## Il problema delle etichette

Il problema è di conti, e si fa in fretta. Un'ora di parlato richiede a un
trascrittore diverse ore di lavoro; le ore di parlato che servirebbero sono
migliaia; e per una lingua parlata da poche persone quel lavoro non lo ha fatto
mai nessuno e non lo farà. Le **etichette**, cioè le risposte giuste scritte da
un umano accanto a ogni esempio, sono la cosa più cara che c'è.

La via d'uscita è spezzare l'apprendimento in due tempi. Prima una lunga fase di
studio su una massa enorme di audio senza risposte, per imparare come è fatto il
suono in generale; poi una breve rifinitura sul compito vero, con le poche
etichette che si hanno. Il primo tempo si chiama **pre-addestramento**
(*pretraining*), il secondo **rifinitura** (*fine-tuning*), e il paradigma non è
nuovo: è lo stesso dei modelli linguistici del {doc}`capitolo sui Transformer </Transformers/overview>`, e
quello che nella visione artificiale si chiama *transfer learning*. Il lavoro
pesante si fa una volta sola, e ogni compito successivo riparte da lì.

`````{tab} Elementare

Un bambino impara i suoni della sua lingua molto prima che qualcuno gli
insegni a scrivere. A furia di ascoltare, si accorge da solo che certi suoni
tornano, che alcune combinazioni sono possibili e altre no, che «pa» e «ba»
sono cose diverse, e tutto questo *senza* che nessuno gli abbia mai mostrato
come si scrivono. Quando poi va a scuola e impara l'alfabeto, parte
avvantaggiato: l'orecchio ha già fatto metà del lavoro.

L'auto-supervisione fa esattamente questo con una macchina. Prima le facciamo
ascoltare montagne di audio senza dirle mai «qui c'è scritto così»: impara da
sola la struttura del suono. Solo dopo le mostriamo poche ore di audio
trascritto, per collegare quella struttura alle parole. Il guadagno è
concreto: le etichette costose servono in quantità molto minore, perché il
grosso (*com'è fatto* il parlato) è già stato imparato gratis.

`````

`````{tab} Superiore

Formalmente disponiamo di un grande insieme di audio non etichettato
$\mathcal{D}_U$ (decine di migliaia di ore) e di un piccolo insieme
etichettato $\mathcal{D}_L = \{(\mathbf{x}^{(i)}, \mathbf{y}^{(i)})\}$, con
$|\mathcal{D}_L| \ll |\mathcal{D}_U|$ (il simbolo $\mathcal{L}$ resta
riservato alle funzioni di perdita). Il pretraining ottimizza su $\mathcal{D}_U$
un obiettivo che non richiede $\mathbf{y}$, un **pretesto** (*pretext task*)
costruito dai dati stessi, per apprendere un encoder $f_\theta$ che mappa la
forma d'onda in rappresentazioni contestuali. Il fine-tuning aggiunge sopra
$f_\theta$ una testa leggera (per l'ASR, tipicamente uno strato con perdita
**CTC**, che vedremo nel capitolo sullo Speech Recognition) e la addestra su
$\mathcal{D}_L$, aggiornando eventualmente anche $\theta$.

Il punto empirico che rende il tutto interessante è la **curva di efficienza
dei dati**: partendo da un encoder pre-addestrato, il WER a valle crolla con
pochissime etichette, là dove un modello addestrato da zero avrebbe bisogno di
ordini di grandezza in più. È la stessa promessa del transfer learning nella
visione, trasferita al dominio del suono.

`````

## wav2vec 2.0: mascherare il suono

Il primo modello a rendere questa idea pienamente convincente per il parlato è
**wav2vec 2.0**, di Alexei Baevski e colleghi a Facebook AI (il laboratorio
che oggi si chiama Meta AI) nel 2020 {cite}`baevski2020wav2vec`. La ricetta
ricalca il «gioco della parola coperta» che nel testo ha reso grande BERT (il
*cloze test*, riempire il buco in una frase) ma applicato a pezzetti di suono
invece che a parole ({numref}`fig-wav2vec-parte-coperta`).

```{figure} ../figures/wav2vec-parte-coperta.svg
:name: fig-wav2vec-parte-coperta
:alt: In alto una forma d'onda spezzata in pezzetti, con due pezzetti coperti da un rettangolo scuro con un punto interrogativo. In basso quattro unità dell'alfabeto sonoro come tessere di un test a risposta multipla, una giusta con la spunta e tre distrattori in grigio; una freccia scende dal pezzetto coperto alla fila delle unità.
:width: 100%

Il gioco della parte coperta: sotto la mascherina c'è una delle unità
dell'alfabeto sonoro, e il modello deve riconoscere quale, fra la giusta e i
distrattori.
```

`````{tab} Elementare

Torna il gioco della frase da completare. Se copro una parola in «Il gatto
nero salta sul ___», tu indovini «muro» perché conosci come funziona la
lingua. wav2vec 2.0 gioca lo stesso gioco con il suono: prende un pezzo di
audio, ne **copre** dei pezzetti, e chiede al modello di indovinare che cosa
c'era sotto.

Ma con un aiuto, perché inventare il suono esatto da zero sarebbe un'impresa
disperata. Il modello ha davanti un piccolo elenco di pezzetti-tipo che si è
costruito lui stesso, una specie di alfabeto sonoro: ognuno si chiama
**unità**. Sotto la parte coperta c'è una di quelle unità, e il gioco è
indovinare **quale**: gli si mette davanti quella giusta insieme a qualche
unità sbagliata (i «distrattori») e deve solo riconoscerla. È un test a
risposta multipla, e per rispondere bene l'orecchio è costretto a capire come
è fatto il parlato.

Una regola tiene in piedi il gioco: le lettere di quell'alfabeto vanno usate
tutte. Se il modello se la cavasse con tre, appiccicate a qualunque suono, sotto
la parte coperta e fra i distrattori finirebbe sempre la stessa cosa, e da un
test con tutte le risposte uguali non si impara niente.

Dopo aver ascoltato in questo modo decine di migliaia di ore di audio senza
etichette, a wav2vec 2.0 bastano appena **dieci minuti** di parlato trascritto
per imparare a riconoscere la voce con una qualità che, solo pochi anni prima,
richiedeva centinaia di ore.

Quel numero però viene raccontato a metà. Il modello che ha imparato ad
ascoltare non ci arriva da solo: accanto a lui lavora un secondo modello, che
non ascolta niente, sa soltanto com'è fatta la lingua e scarta le parole
improbabili. Il risultato è della coppia. Ascoltare montagne di audio risolve
il problema di quante trascrizioni servono; non insegna l'italiano.

`````

`````{tab} Superiore

L'architettura ha tre stadi. Un **encoder convoluzionale** $f$ trasforma la
forma d'onda grezza $\mathbf{x}$ in una sequenza di vettori latenti
$\mathbf{Z} = (\mathbf{z}_1, \dots, \mathbf{z}_T)$, uno ogni ~20 ms. Un
**Transformer** $g$ legge $\mathbf{Z}$
(con alcuni tratti mascherati) e produce rappresentazioni *contestuali*
$\mathbf{C} = (\mathbf{c}_1, \dots, \mathbf{c}_T)$, in cui ogni $\mathbf{c}_t$
tiene conto dell'intera frase. In
parallelo, un modulo di **quantizzazione** sostituisce ogni $\mathbf{z}_t$ con una voce
di un piccolo dizionario appreso (in gergo, *product quantization*: due codebook
da 320 voci ciascuno, e le due voci scelte si concatenano), producendo il
bersaglio discreto $\mathbf{q}_t$: è il modo di darsi un «alfabeto» finito di unità di
suono senza definirlo a mano.

Attenzione a *come* avviene la scelta, perché non è quella che verrà definita
nella sezione sui codec neurali: qui non si cerca l'entrata più vicina in
distanza. $\mathbf{z}_t$ viene proiettato su una griglia di $G \times V$
**logit** (con $G = 2$ gruppi e $V = 320$ voci per gruppo) e l'indice
è l’`argmax` della **Gumbel-softmax** di quei logit, cioè della softmax dei
logit perturbati con rumore di Gumbel e temperatura $\tau$; all'indietro si usa
lo *straight-through*, che rende derivabile una scelta discreta. La differenza
non è di dettaglio: senza logit non ci sarebbe niente da regolarizzare, ed è
proprio la ragione per cui serve il termine di diversità.

L'obiettivo è **contrastivo**. Per ogni passo mascherato $t$, dato il vettore
contestuale $\mathbf{c}_t$, il modello deve riconoscere la vera unità quantizzata
$\mathbf{q}_t$
in mezzo a un insieme $\mathcal{Q}_t$ formato da $\mathbf{q}_t$ e da $K$
distrattori (nel paper,
$K = 100$) pescati da altri passi mascherati:

$$
\mathcal{L}_m = -\log
\frac{\exp\!\big(\mathrm{sim}(\mathbf{c}_t, \mathbf{q}_t)/\kappa\big)}
{\sum_{\tilde{\mathbf{q}}\,\in\,\mathcal{Q}_t}\exp\!\big(\mathrm{sim}(\mathbf{c}_t, \tilde{\mathbf{q}})/\kappa\big)},
$$

dove $\mathrm{sim}(\mathbf{a},\mathbf{b})$ è la **similarità del coseno** (già usata per gli
embedding), $\kappa$ una temperatura e $\mathcal{Q}_t$ l'insieme dei candidati. È una
softmax che premia il modello quando assegna a $\mathbf{q}_t$ la probabilità più alta.
Un secondo termine di **diversità** incoraggia a usare tutte le voci del
dizionario, evitando che ne collassi solo qualcuna. A valle, con una testa CTC
su pochissime etichette, wav2vec 2.0 raggiunge un WER di
$4{,}8/8{,}2$ su *test-clean/test-other* di Librispeech usando **10 minuti**
di dati etichettati e 53.000 ore non etichettate in pre-addestramento
{cite}`baevski2020wav2vec`.

Quel numero però va letto per intero, perché è la cifra più citata del paper ed
è quasi sempre citata male: è ottenuto **decodificando con un modello di lingua
Transformer**. Il solo modello acustico, nella stessa configurazione, sta a
$40{,}2/38{,}7$ (la tabella in appendice del paper smonta il contributo della
decodifica); con un modello di lingua a 4-grammi si passa
a $6{,}6/10{,}3$, e solo con quello Transformer si arriva a $4{,}8/8{,}2$. Fra il
primo e l'ultimo l'errore si divide per otto sul test pulito e per quasi cinque
su quello difficile. Il pre-addestramento risolve il problema
delle **etichette acustiche**, non sostituisce il modello di lingua: è una
distinzione che il capitolo sullo Speech Recognition riprenderà pari pari,
quando metterà in fila i pezzi di una pipeline di riconoscimento.

`````

Il cuore del compito si vede in poche righe di NumPy, e la domanda a cui
risponde è semplice: il modello punta sull'unità giusta o su uno dei
distrattori? Ogni candidato è un gruppetto di numeri. Per misurare quanto
somiglia a ciò che il modello si è fatto in mente del pezzetto coperto si
calcola un solo numero, che vale 1 quando i due gruppetti dicono la stessa cosa
e 0 quando non hanno niente in comune: si chiama **similarità del coseno**. I
cinque punteggi vengono poi riscalati in modo che sommino a uno, così si leggono
come probabilità: quanta fiducia il modello mette su ciascun candidato.

```python
import numpy as np

rng = np.random.default_rng(0)
d = 8  # quanti numeri ha ogni gruppetto (nei modelli veri sono centinaia)

# c: quello che il modello si e' fatto in mente del pezzetto COPERTO.
# In un modello ben addestrato e' vicino all'unita' giusta e lontano dai distrattori.
c = rng.standard_normal(d)

# q_true: l'unita' corretta (qui una versione "vicina" a c);
# i distrattori sono unita' pescate da altri pezzetti coperti della stessa frase.
q_true = c + 0.3 * rng.standard_normal(d)
q_dist = rng.standard_normal((4, d))
candidati = np.vstack([q_true, q_dist])      # (5, d): il vero piu' 4 distrattori

def coseno(a, B):                            # coseno tra a e ogni riga di B
    a = a / np.linalg.norm(a)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    return B @ a

kappa = 0.1                                  # "temperatura": piu' e' bassa,
                                             # piu' la scelta esce netta
punteggi = coseno(c, candidati) / kappa
prob = np.exp(punteggi - punteggi.max())
prob /= prob.sum()                           # softmax: i punteggi riscalati
                                             # cosi' che sommino a uno

print("prob. per candidato:", prob.round(3))
print("scelto:", int(prob.argmax()), "(0 = unita' giusta)")
```

```text
prob. per candidato: [0.672 0.    0.    0.271 0.056]
scelto: 0 (0 = unita' giusta)
```

L'unità giusta vince, ma il margine merita uno sguardo: si prende due terzi
della probabilità, e un distrattore pescato a caso se ne prende un quarto. Non è
un difetto del conto: è che i gruppetti di numeri, qui, sono cortissimi.

Vale una regola che tornerà spesso: più numeri ha un gruppetto, più è difficile
che due gruppetti pescati a caso si somiglino. Con pochi numeri, invece, succede
in continuazione, per puro accidente. Qui ne abbiamo otto, che è pochissimo, ed
è il motivo per cui un distrattore preso a caso finisce quasi addosso al
bersaglio.

Nel modello vero i numeri sono centinaia, e allora due gruppetti a caso non si
somigliano quasi mai: un distrattore qualunque si scarterebbe senza fatica. Il
compito resta difficile lo stesso, ma per un'altra ragione, e conviene tenerla
presente. I distrattori veri non sono presi a caso: sono unità pescate da altri
punti coperti della *stessa* frase, quindi suoni imparentati con quello da
indovinare. E sono cento, non quattro.

## HuBERT: darsi le etichette da soli

Il gioco di wav2vec 2.0 funziona, ma poggia tutto su una cosa delicata:
l'elenco di pezzetti-tipo in mezzo a cui il modello deve riconoscere quello
giusto. Se quell'elenco è fatto male il gioco premia la risposta sbagliata, e
costruirlo bene *mentre* lo si sta già usando è tutt'altro che semplice.
Nel 2021 Wei-Ning Hsu e colleghi, sempre a Facebook AI, propongono con **HuBERT**
(*Hidden-Unit BERT*) una strada diversa e sorprendentemente semplice
{cite}`hsu2021hubert`: invece di riconoscere l'unità giusta tra distrattori,
darsi da soli delle **pseudo-etichette** e imparare a predirle, esattamente
come BERT predice la parola mascherata.

`````{tab} Elementare

Come si scrive una lingua che non ha alfabeto? Te ne inventi uno provvisorio:
raggruppi i suoni che ti sembrano simili e dai a ogni gruppo un simbolo («suono
numero 1», «suono numero 2»), etichette rozze, inventate da te. Poi giochi al
solito gioco della parola coperta: nascondi dei tratti di audio e ti alleni a
indovinare *quale simbolo* c'era sotto. Il bello arriva dopo: quando ti sei
fatto l'orecchio, i tuoi raggruppamenti diventano più sensati di quelli di
partenza, e allora rifai l'alfabeto con quelli e ricominci. Un ciclo che si
affina da solo, come uno schizzo ripassato più volte a matita finché il disegno
emerge.

Non serve che l'alfabeto sia giusto: serve che sia **coerente**, cioè che dia
lo stesso simbolo a suoni davvero simili. Se la stessa «sss» finisse ora sotto
un simbolo ora sotto un altro, a caso, ti alleneresti a indovinare
l'imprevedibile e non ne verrebbe fuori niente. Quando invece i gruppi tengono,
azzeccare il simbolo coperto costringe a capire come è fatto il suono, anche se
quei nomi te li sei inventati tu.

`````

`````{tab} Superiore

HuBERT alterna due passi. **Passo di clustering** (offline): si estraggono
feature dall'audio e le si raggruppa con un semplice **k-means**, ottenendo
per ogni frame un'etichetta discreta $u_t \in \{1, \dots, V\}$, l’«unità
nascosta», dove $V$ è il numero di cluster, cioè la taglia dell'inventario
discreto: lo stesso ruolo che $V$ ha in wav2vec 2.0, dove conta le voci di un
singolo codebook. La lettera è diversa apposta rispetto a $\mathbf{z}_t$: qui
$u_t$ è un **intero**, un nome di gruppo, mentre lo $\mathbf{z}_t$ di wav2vec
2.0 è un **vettore** di numeri reali. È la differenza di fondo fra i due
metodi, e si vede nei simboli. Nella prima iterazione le feature sono banali
MFCC; nelle successive si usano le rappresentazioni interne del HuBERT già
addestrato, che danno cluster via via migliori. **Passo di predizione mascherata**: si
maschera un sottoinsieme $M$ di frame e si addestra il modello, alla BERT, a
predire le pseudo-etichette dei frame mascherati:

$$
\mathcal{L} = -\sum_{t \,\in\, M} \log\, p_\theta\!\left(u_t \mid \tilde{\mathbf{X}}, t\right),
$$

dove $\tilde{\mathbf{X}}$ è la sequenza di frame con i tratti mascherati, $u_t$ l'unità nascosta
assegnata dal k-means al frame $t$ e $p_\theta$ la distribuzione, prodotta dal
modello, sulle $V$ unità del dizionario. È una normale cross-entropia su un
problema di classificazione, senza distrattori né obiettivo contrastivo.

Perché funziona pur partendo da etichette rozze? Perché ciò che conta non è la
*correttezza* del clustering ma la sua **coerenza**: se il k-means assegna lo
stesso simbolo a frame acusticamente simili, predire quel simbolo forza il
modello a modellare la struttura del segnale. E l'iterazione chiude il cerchio:
rappresentazioni migliori $\to$ cluster migliori $\to$ bersagli migliori. A
parità di condizioni, HuBERT eguaglia o supera wav2vec 2.0 nei regimi a basse
risorse su Librispeech {cite}`hsu2021hubert`.

`````

Conviene notare la parentela, perché i due si somigliano più di quanto sembri.
Dentro sono fatti allo stesso modo, e il motore è identico: coprire dei pezzi
di audio e costringere il modello a tirare fuori quello che c'era sotto.
Cambia il **bersaglio**, cioè che cosa esattamente gli si chiede di
indovinare. Uno gliela fa riconoscere in mezzo a dei distrattori, come in un
test a crocette; l'altro gli chiede di dirne il nome, e i nomi possibili sono
quelli dell'alfabeto provvisorio.

E cambia anche *quando* il bersaglio viene deciso. wav2vec 2.0 se lo costruisce
mentre impara, con lo stesso addestramento che poi lo deve indovinare: l'elenco
si muove sotto i piedi del gioco. HuBERT invece se lo prepara a parte, prima di
cominciare, e lo rifà solo ogni tanto: mentre si gioca, l'elenco sta fermo.

## A cosa servono

Queste rappresentazioni pre-addestrate sono diventate un **mattone** di buona
parte dei sistemi audio moderni. Il caso di scuola è il riconoscimento vocale,
che d'ora in poi chiameremo con la sua sigla inglese, **ASR** (*automatic speech
recognition*), la stessa che userà per intero il capitolo successivo.

Serve soprattutto dove le trascrizioni scarseggiano: una lingua parlata da poche
persone, un dialetto, un mestiere di cui nessuno ha mai raccolto registrazioni
annotate. Si parte da un modello che ha ascoltato montagne di audio senza
etichette, gli si mostrano le poche ore trascritte che esistono, e si arriva a
risultati che prima erano impensabili. Il collegamento con il capitolo sullo
Speech Recognition è diretto: la parte che lì trasformerà le rappresentazioni in
parole non fa che rifinire ciò che il pre-addestramento ha già preparato.

Gli stessi numeri servono poi a classificare suoni ambientali, a identificare
chi parla, a riconoscere emozioni o lingua, e (come vedremo nelle prossime due
sezioni) fanno da punto di partenza anche per la **generazione** di audio: i
pezzetti-tipo imparati qui diventano un vocabolario su cui un modello può
«scrivere» suono, come un modello linguistico scrive testo.

Onestà d'obbligo, però, sui limiti. Queste rappresentazioni colgono benissimo
i suoni e come si incastrano fra loro: quali sono, in che ordine, con che
timbro. È esattamente ciò che il gioco della parte coperta premia, e non c'è da
stupirsi.

Colgono molto meno il **significato**. Un modello addestrato così sa che due
frammenti suonano simili, non sa se una frase è ironica o se una domanda vuole
una certa risposta. È un orecchio finissimo, non una mente che comprende. Il
senso lo mettono i pezzi che vengono dopo, che su questo orecchio si appoggiano.
Il pre-addestramento audio risolve il problema delle etichette, non quello della
comprensione: distinguere le due cose è il primo passo per usarlo bene.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Trascrivere l'audio costa; di audio **non trascritto** ce n'è a valanga. Un
  modello può imparare da solo com'è fatto il suono (come il bambino che sente
  la lingua prima di saperla scrivere) e solo dopo imparare il compito vero con
  **poche** ore di esempi corretti.
- Quello che impara si chiama **rappresentazione**: il gruppetto di numeri con
  cui si tiene in mente un pezzetto di suono, fatto in modo che pezzetti simili
  finiscano vicini.
- **wav2vec 2.0** gioca al gioco della parola coperta: nasconde dei tratti di
  audio e chiede di **riconoscere** quello giusto in mezzo a qualche
  distrattore, come un test a crocette. Con dieci minuti di parlato trascritto
  arriva dove prima servivano centinaia di ore, purché ad aiutarlo ci sia anche
  un modello che sa com'è fatta la lingua.
- **HuBERT** cambia gioco: si inventa un alfabeto provvisorio raggruppando i
  suoni che si somigliano, poi si allena a indovinare *quale simbolo* stava
  sotto la parte coperta, e ogni tanto rifà l'alfabeto meglio di prima. Non
  serve che sia giusto, serve che sia **coerente**.
- Questo orecchio è finissimo ma non è una mente: sa che due frammenti suonano
  simili, non sa se una frase è ironica. Il significato lo mettono i pezzi che
  vengono dopo.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Etichettare l'audio è costoso, ma di audio **non etichettato** ce n'è a
  valanga: l'apprendimento **auto-supervisionato** impara la struttura del
  suono da solo (pretraining), poi rifinisce sul compito vero con **poche
  etichette** (fine-tuning) (lo stesso salto che nel testo va da word2vec a
  BERT).
- **wav2vec 2.0** {cite}`baevski2020wav2vec`: encoder convoluzionale +
  Transformer, i latenti sono **quantizzati** in unità discrete, e mascherando
  parti del segnale il modello impara con un obiettivo **contrastivo** a
  riconoscere l'unità giusta tra distrattori (il *cloze test* del suono). La
  scelta dell'unità passa per una **Gumbel-softmax sui logit**, non per la
  distanza dal prototipo più vicino. Con 10 minuti di etichette raggiunge un
  WER di $4{,}8/8{,}2$ su Librispeech, ma **con un modello di lingua
  Transformer** in decodifica: il solo modello acustico sta intorno al 40 %.
- **HuBERT** {cite}`hsu2021hubert`: niente contrastivo, ma **pseudo-etichette**
  da un **k-means** (unità nascoste) predette sui frame mascherati, con
  **iterazione** che raffina i cluster. Conta la *coerenza* dei bersagli, non
  la loro correttezza.
- Le due condividono l'ossatura e differiscono nel bersaglio: **riconoscere**
  (contrastivo) contro **predire** una classe (masked prediction).
- Queste rappresentazioni sono il mattone di ASR a basse risorse,
  classificazione audio e generazione; catturano bene **fonetica e struttura**,
  molto meno il **significato** ad alto livello.
```

`````
