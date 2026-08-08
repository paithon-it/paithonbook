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
etichette? È l'idea dell'apprendimento **auto-supervisionato** (*self-
supervised*), e nell'audio ha prodotto negli ultimi anni un salto paragonabile
a quello che, nel testo, ha portato dai word embedding come word2vec (visti
nella sezione sugli embedding) ai grandi modelli pre-addestrati come BERT: da
rappresentazioni fisse a rappresentazioni imparate dal contesto,
riutilizzabili ovunque.

Diamo per acquisite le rappresentazioni di base del suono (campionamento,
spettrogramma, scala mel, MFCC) che abbiamo costruito nella sezione *Dal suono
alle feature*: qui non ripartiamo dall'onda di pressione, ma da come un
modello impara *da sé* una rappresentazione migliore.

## Il problema delle etichette

Il paradigma è lo stesso incontrato nel capitolo sui Transformer con il
**pretraining** dei modelli linguistici, e nella visione artificiale con il
**transfer learning**: prima una lunga fase di studio senza etichette, su una
mole enorme di dati grezzi, per imparare rappresentazioni generali; poi una
breve fase di rifinitura (*fine-tuning*) sul compito specifico, con i pochi
dati etichettati che si hanno. Il pre-addestramento fa il lavoro pesante una
volta sola; a valle, ogni compito riparte da lì.

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
etichettato $\mathcal{D}_L = \{(X^{(i)}, \mathbf{y}^{(i)})\}$, con
$|\mathcal{D}_L| \ll |\mathcal{D}_U|$ (il simbolo $\mathcal{L}$ resta
riservato, come nel resto del libro, alle funzioni di perdita che incontreremo
tra poco). Il pretraining ottimizza su $\mathcal{D}_U$
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
invece che a parole.

`````{tab} Elementare

Torna il gioco della frase da completare. Se copro una parola in «Il gatto
nero salta sul ___», tu indovini «muro» perché conosci come funziona la
lingua. wav2vec 2.0 gioca lo stesso gioco con il suono: prende un pezzo di
audio, ne **nasconde** dei tratti, e chiede al modello di indovinare che cosa
c'era sotto. Ma con un aiuto: non deve inventare il suono esatto da zero
(sarebbe un'impresa), gli si mette davanti l'unità giusta insieme a qualche
unità sbagliata (dei «distrattori») e deve solo **riconoscere** quale è quella
buona. Come un test a risposta multipla in cui l'orecchio, per rispondere
bene, è costretto a capire come è fatto il parlato.

La cosa notevole è quanto rende. Dopo aver ascoltato in questo modo decine di
migliaia di ore di audio senza etichette, a wav2vec 2.0 bastano appena **dieci
minuti** di parlato trascritto per imparare a riconoscere la voce con una
qualità che, solo pochi anni prima, richiedeva centinaia di ore.

`````

`````{tab} Superiore

L'architettura ha tre stadi. Un **encoder convoluzionale** $f$ trasforma la
forma d'onda grezza $X$ in una sequenza di vettori latenti
$Z = (z_1, \dots, z_T)$, uno ogni ~20 ms. Un **Transformer** $g$ legge $Z$
(con alcuni tratti mascherati) e produce rappresentazioni *contestuali*
$C = (c_1, \dots, c_T)$, in cui ogni $c_t$ tiene conto dell'intera frase. In
parallelo, un modulo di **quantizzazione** discretizza ogni $z_t$ nell'entrata
più vicina di un piccolo dizionario appreso (in gergo, *product quantization*
con due codebook da 320 voci), producendo il bersaglio discreto $q_t$: è il
modo di darsi un «alfabeto» finito di unità di suono senza definirlo a mano.

L'obiettivo è **contrastivo**. Per ogni passo mascherato $t$, dato il vettore
contestuale $c_t$, il modello deve riconoscere la vera unità quantizzata $q_t$
in mezzo a un insieme $Q_t$ formato da $q_t$ e da $K$ distrattori (nel paper,
$K = 100$) pescati da altri passi mascherati:

$$
\mathcal{L}_m = -\log
\frac{\exp\!\big(\mathrm{sim}(c_t, q_t)/\kappa\big)}
{\sum_{\tilde{q}\,\in\,Q_t}\exp\!\big(\mathrm{sim}(c_t, \tilde{q})/\kappa\big)},
$$

dove $\mathrm{sim}(a,b)$ è la **similarità del coseno** (già usata per gli
embedding), $\kappa$ una temperatura e $Q_t$ l'insieme dei candidati. È una
softmax che premia il modello quando assegna a $q_t$ la probabilità più alta.
Un secondo termine di **diversità** incoraggia a usare tutte le voci del
dizionario, evitando che ne collassi solo qualcuna. A valle, con la sola
perdita CTC su pochissime etichette, wav2vec 2.0 raggiunge un WER di
$4{,}8/8{,}2$ su *test-clean/test-other* di Librispeech usando **10 minuti**
di dati etichettati e 53.000 ore non etichettate in pre-addestramento
{cite}`baevski2020wav2vec`.

`````

Il cuore del compito contrastivo si vede in poche righe di NumPy: dato il
vettore contestuale del frame mascherato, calcoliamo la similarità del coseno
verso l'unità giusta e i distrattori, e trasformiamo i punteggi in una
distribuzione con la softmax. Un modello ben addestrato concentra la
probabilità sull'unità corretta.

```python
import numpy as np

rng = np.random.default_rng(0)
d = 8  # dimensione dei vettori latenti

# c: rappresentazione contestuale del frame MASCHERATO, prodotta dal Transformer.
# In un modello ben addestrato e' vicina all'unita' giusta e lontana dai distrattori.
c = rng.standard_normal(d)

# q_true: l'unita' quantizzata corretta (qui una versione "vicina" a c);
# i distrattori sono unita' pescate da altri frame mascherati della stessa frase.
q_true = c + 0.3 * rng.standard_normal(d)
q_dist = rng.standard_normal((4, d))
candidati = np.vstack([q_true, q_dist])      # (5, d): il vero piu' 4 distrattori

def coseno(a, B):                            # coseno tra a e ogni riga di B
    a = a / np.linalg.norm(a)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    return B @ a

kappa = 0.1                                  # temperatura
punteggi = coseno(c, candidati) / kappa
prob = np.exp(punteggi - punteggi.max())
prob /= prob.sum()                           # softmax sui candidati

print("prob. per candidato:", prob.round(3))
print("scelto:", int(prob.argmax()), "(0 = unita' giusta)")
```

```text
prob. per candidato: [0.672 0.    0.    0.271 0.056]
scelto: 0 (0 = unita' giusta)
```

## HuBERT: darsi le etichette da soli

L'obiettivo contrastivo di wav2vec 2.0 funziona, ma ha una fragilità: dipende
da quanto sono «buoni» i bersagli quantizzati, e definirli bene è delicato.
Nel 2021 Wei-Ning Hsu e colleghi, sempre a Facebook AI, propongono con **HuBERT**
(*Hidden-Unit BERT*) una strada diversa e sorprendentemente semplice
{cite}`hsu2021hubert`: invece di riconoscere l'unità giusta tra distrattori,
darsi da soli delle **pseudo-etichette** e imparare a predirle, esattamente
come BERT predice la parola mascherata.

`````{tab} Elementare

Immagina di dover imparare a scrivere una lingua di cui nessuno conosce
l'alfabeto. Cosa fai? Te ne inventi uno provvisorio: raggruppi i suoni che ti
sembrano simili e dai a ogni gruppo un simbolo («suono numero 1», «suono
numero 2») anche se sono etichette rozze, inventate da te. Poi giochi al
solito gioco della parola coperta: nascondi dei tratti di audio e ti alleni a
indovinare *quale simbolo* c'era sotto. Il bello arriva dopo: una volta che il
modello ha imparato un po', i suoi raggruppamenti diventano più sensati di
quelli di partenza, e allora rifai l'alfabeto usando i suoi (più fine del
primo) e ricominci. Un ciclo che si affina da solo, come uno schizzo ripassato
più volte a matita finché il disegno emerge.

Il trucco furbo è che l'alfabeto *non deve essere perfetto*: deve solo essere
**coerente**, cioè assegnare lo stesso simbolo a suoni davvero simili. Anche se
i nomi dei gruppi sono arbitrari, imparare a predirli costringe comunque il
modello a capire la struttura del suono.

`````

`````{tab} Superiore

HuBERT alterna due passi. **Passo di clustering** (offline): si estraggono
feature dall'audio e le si raggruppa con un semplice **k-means**, ottenendo
per ogni frame un'etichetta discreta $z_t \in \{1, \dots, C\}$ (un'«unità
nascosta»). Nella prima iterazione le feature sono banali MFCC; nelle
successive si usano le rappresentazioni interne del HuBERT già addestrato, che
danno cluster via via migliori. **Passo di predizione mascherata**: si
maschera un sottoinsieme $M$ di frame e si addestra il modello, alla BERT, a
predire le pseudo-etichette dei frame mascherati:

$$
\mathcal{L} = -\sum_{t \,\in\, M} \log\, p_\theta\!\left(z_t \mid \tilde{X}, t\right),
$$

dove $\tilde{X}$ è la sequenza con i tratti mascherati, $z_t$ l'unità nascosta
assegnata dal k-means al frame $t$ e $p_\theta$ la distribuzione, prodotta dal
modello, sulle $C$ unità del dizionario. È una normale cross-entropia su un
problema di classificazione, senza distrattori né obiettivo contrastivo.

Perché funziona pur partendo da etichette rozze? Perché ciò che conta non è la
*correttezza* del clustering ma la sua **coerenza**: se il k-means assegna lo
stesso simbolo a frame acusticamente simili, predire quel simbolo forza il
modello a modellare la struttura del segnale. E l'iterazione chiude il cerchio:
rappresentazioni migliori $\to$ cluster migliori $\to$ bersagli migliori. A
parità di condizioni, HuBERT eguaglia o supera wav2vec 2.0 nei regimi a basse
risorse su Librispeech {cite}`hsu2021hubert`.

`````

Vale la pena notare la parentela: wav2vec 2.0 e HuBERT condividono la stessa
ossatura (encoder convoluzionale più Transformer, con il mascheramento come
motore dell'apprendimento) e differiscono nel *bersaglio*. Uno lo riconosce
tra distrattori (contrastivo), l'altro lo predice come una classe (predizione
mascherata su unità date da un clustering iterativo). Due risposte alla stessa
domanda: quale «alfabeto del suono» far indovinare al modello, e come
definirlo senza etichette umane.

## A cosa servono

Queste rappresentazioni pre-addestrate sono diventate un **mattone** di buona
parte dei sistemi audio moderni. Il caso di scuola è il riconoscimento vocale
a **basse risorse**: lingue e dialetti per cui esistono poche ore trascritte
partono da un encoder addestrato su tanto audio non etichettato e raggiungono
prestazioni prima impensabili; il collegamento diretto con la pipeline ASR del
capitolo sullo Speech Recognition, dove la testa CTC o il decoder con
attenzione si limiterà a rifinire ciò che il pretraining ha già preparato. Ma
gli stessi vettori servono a classificare suoni ambientali, identificare chi
parla, riconoscere emozioni o lingua, e (come vedremo) fanno da
rappresentazione di partenza anche per la **generazione** di audio, dove le
unità discrete imparate qui diventano un vocabolario su cui un modello può
«scrivere» suono come un modello linguistico scrive testo.

Onestà d'obbligo, però, sui limiti. Queste rappresentazioni catturano molto
bene la **fonetica** e la struttura locale del segnale (quali suoni, come si
concatenano, con quale timbro), perché è esattamente ciò che il compito di
mascheramento premia. Catturano molto meno il **significato** ad alto livello:
un encoder auto-supervisionato «sa» che due frammenti suonano simili, non che
una frase è ironica o che una domanda richiede una certa risposta. È un
orecchio finissimo, non una mente che comprende. Per arrivare al senso servono
i moduli a valle (un decoder linguistico, un modello di dialogo), che su
questo orecchio si appoggiano. Il pretraining audio risolve il problema delle
etichette, non quello della comprensione: distinguere le due cose è il primo
passo per usarlo bene.

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
  riconoscere l'unità giusta tra distrattori (il *cloze test* del suono). Con
  10 minuti di etichette raggiunge ottime prestazioni ASR.
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
