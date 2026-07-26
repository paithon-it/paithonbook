# Un'etichetta per ogni parola: POS tagging e riconoscimento di entità

«La vecchia porta la sbarra». Leggi la frase una prima volta: un'anziana
signora trasporta una sbarra di ferro. Ora rileggila: una porta malandata
blocca il passaggio a qualcuna. Nessun trucco di punteggiatura: le stesse
cinque parole, nello stesso ordine, formano due frasi italiane complete e
sensate. Il bivio è tutto grammaticale — «vecchia» può essere nome o
aggettivo, «porta» nome o verbo, «la» articolo o pronome, «sbarra» nome o
verbo — e scegliere una lettura significa, senza accorgersene, assegnare a
ogni parola il suo ruolo nella frase.

Noi lo facciamo in una frazione di secondo. Una macchina
deve farlo *esplicitamente*: scrivere accanto a ogni parola un'etichetta con
il suo ruolo. È il **part-of-speech tagging** (POS, etichettatura delle
parti del discorso), uno dei compiti più antichi del NLP; suo cugino stretto
è il **riconoscimento di entità nominate** (NER), già incontrato nella
panoramica del capitolo. Li raccontiamo insieme perché condividono la stessa
forma — un'etichetta per ogni parola — e la stessa storia: prima i modelli
probabilistici degli anni Novanta, la «seconda stagione» della parabola
storica del capitolo, poi le reti ricorrenti delle sezioni precedenti.

## Il mestiere di ogni parola

A scuola si chiamava analisi grammaticale: articolo, nome, verbo, aggettivo…
Il POS tagging è la stessa cosa, fatta da un algoritmo su milioni di frasi.
Per l'esempio ricorrente del libro il risultato è:

> Il/`DET` gatto/`NOUN` nero/`ADJ` salta/`VERB` su/`ADP` il/`DET` muro/`NOUN`

(nota «sul», che l'italiano fonde e che l'annotazione scompone in
preposizione più articolo). Perché il gioco funzioni tra lingue diverse
serve però un inventario di categorie condiviso: è il contributo del
progetto **Universal Dependencies** {cite}`nivre2016universal`, nato per
annotare molte lingue — 33 alla presentazione del 2016, oggi quasi duecento —
con gli stessi criteri.

`````{tab} Elementare

Pensa alle categorie grammaticali come ai *mestieri* delle parole: il nome
indica cose e persone, il verbo racconta azioni, l'articolo fa strada al
nome. Il punto delicato è che molte parole fanno due mestieri, e cambiano
divisa senza avvisare: «porta» è un oggetto in «la porta cigola» e un'azione
in «Maria porta il pane»; «ancora» è un pezzo di nave se la pronunci
*àncora* e un avverbio se la pronunci *ancóra*. Sulla pagina le due «ancora»
sono identiche: solo le parole intorno rivelano quale hai davanti.
Etichettare le parti del discorso è proprio questo: guardare il contesto e
decidere, parola per parola, quale mestiere è in servizio. I linguisti del
progetto Universal Dependencies hanno stilato una lista di 17 mestieri che
funziona per l'italiano come per il finlandese o il giapponese — una specie
di stele di Rosetta della grammatica.

`````

`````{tab} Superiore

Formalmente il POS tagging è un problema di **etichettatura di sequenze**:
data la frase $w_1, \dots, w_n$, produrre la sequenza di etichette
$t_1, \dots, t_n$, una per token, dalla stessa lunghezza dell'input. È una
struttura più semplice della traduzione vista nella sezione precedente
(niente riordini, niente lunghezze diverse), ma la difficoltà si concentra
nell'ambiguità: le parole ambigue sono una minoranza del vocabolario, però
sono tra le più frequenti, tanto che in un testo inglese corrente oltre la
metà delle occorrenze ammette più di un'etichetta.

Lo standard di riferimento è il tagset **universale** di Universal
Dependencies {cite}`nivre2016universal`, 17 categorie valide per tutte le
lingue del progetto:

| Classi aperte | Classi chiuse | Altro |
|---|---|---|
| `NOUN` nome | `DET` determinante | `PUNCT` punteggiatura |
| `PROPN` nome proprio | `PRON` pronome | `SYM` simbolo |
| `VERB` verbo | `ADP` adposizione | `X` altro |
| `ADJ` aggettivo | `AUX` ausiliare | |
| `ADV` avverbio | `CCONJ` cong. coordinante | |
| `INTJ` interiezione | `SCONJ` cong. subordinante | |
| | `NUM` numerale | |
| | `PART` particella | |

Le classi *aperte* accolgono parole nuove di continuo («googlare» è un
`VERB` recente), quelle *chiuse* quasi mai: nessuno conia nuovi articoli.
Sull'inglese giornalistico i tagger moderni superano il 97% di accuratezza
per token — un numero da leggere con prudenza, come vedremo parlando di
valutazione.

`````

E a che cosa serve, oggi, un'etichetta grammaticale? Alla
**lemmatizzazione**: per ricondurre «porta» al suo lemma devi sapere se è il
nome (lemma *porta*) o il verbo (lemma *portare*). Alla **sintesi vocale**,
il percorso inverso del riconoscimento vocale che incontreremo nel capitolo
dedicato alla voce: un lettore automatico davanti ad «ancora» deve scegliere
tra *àncora* e *ancóra*, e l'accento giusto lo decide la categoria
grammaticale. E all'**analisi sintattica**: le etichette POS sono i mattoni
con cui, nella prossima sezione, si costruisce l'impalcatura della frase.

## Chi, dove, quando: le entità nominate

Il secondo compito lo abbiamo già visto all'opera nella panoramica del
capitolo: in «Enrico Fermi nacque a Roma nel 1901» un sistema NER etichetta
*Enrico Fermi* come persona, *Roma* come luogo, *1901* come data. Il
**riconoscimento di entità nominate** — la sigla nasce alle *Message
Understanding Conference* degli anni Novanta — cerca nel testo persone,
luoghi e organizzazioni, più date, cifre e importi: è il primo passo per
popolare una base di conoscenza, anonimizzare una cartella clinica,
collegare un articolo ai personaggi che cita.

A prima vista sembra un problema diverso dal POS tagging: lì un'etichetta
per parola, qui *segmenti* da ritagliare — «Enrico Fermi» è un'entità sola,
lunga due parole. Il trucco che riporta tutto alla forma già nota è lo
**schema BIO**, proposto negli anni Novanta (Ramshaw e Marshall, 1995).

`````{tab} Elementare

Immagina di lavorare con degli evidenziatori colorati: giallo per le
persone, azzurro per i luoghi, verde per le date. Il problema è dettare al
telefono, parola per parola, dove passa l'evidenziatore — e con quale
regola? Ne bastano tre per parola: «qui **comincio** un'evidenziatura
gialla», «qui la **continuo**», «qui la penna è **sollevata**». Sulla frase
di Fermi: *Enrico* = comincio-giallo, *Fermi* = continuo, *nacque, a* =
penna su, *Roma* = comincio-azzurro, *nel* = penna su, *1901* =
comincio-verde. La distinzione tra «comincio» e «continuo» sembra pignola
ma è preziosa: in «il faccia a faccia Mattarella Macron», due
«comincio-giallo» di fila dicono che le persone sono *due*; un «comincio»
seguito da un «continuo» direbbe che è una sola, un improbabile signor
Mattarella Macron.

`````

`````{tab} Superiore

Lo schema BIO trasforma l'estrazione di segmenti in etichettatura per
token. Per ogni tipo di entità $X$ si definiscono due etichette — `B-X`
(*begin*, primo token del segmento) e `I-X` (*inside*, continuazione) — più
un'unica etichetta `O` (*outside*) per i token fuori da ogni entità: con
$K$ tipi, il tagset conta $2K + 1$ etichette. La frase di Fermi diventa:

> Enrico/`B-PER` Fermi/`I-PER` nacque/`O` a/`O` Roma/`B-LOC` nel/`O`
> 1901/`B-DATE`

La marca `B` è ciò che rende lo schema invertibile: senza di essa due
entità adiacenti dello stesso tipo si fonderebbero in una. La sequenza
`B-PER B-PER` codifica due persone consecutive; `B-PER I-PER` una sola
entità di due token. Esistono varianti più ricche (BIOES aggiunge etichette
esplicite di fine segmento e di entità a token singolo), ma l'idea non
cambia: una volta ridotto il NER a un'etichetta per token, *qualunque*
modello di etichettatura di sequenze — HMM, CRF, BiLSTM, Transformer — lo
può affrontare.

`````

## La grammatica dietro la tenda: gli HMM

Come si insegna a una macchina a etichettare? La risposta classica, cuore
della stagione statistica del NLP, è un modello dal nome intimidatorio e
dall'idea limpida: lo **Hidden Markov Model** (HMM, modello di Markov
nascosto). L'aggettivo importante è *nascosto*: le categorie grammaticali
non si vedono mai — sulla pagina ci sono solo parole — eppure sono loro a
governare quali parole compaiono e in che ordine.

`````{tab} Elementare

Immagina una recita dietro una tenda: tu, in platea, non vedi gli attori,
senti solo le battute. Gli attori sono le *categorie grammaticali*, che si
passano la scena secondo abitudini precise (dopo l'ARTICOLO entra quasi
sempre il NOME, diciamo 7 volte su 10, e raramente il VERBO), e ognuna ha il
suo copione di parole tipiche (quando è in scena l'ARTICOLO senti «la»,
«il», «un»…). Un HMM è questo teatro: due libretti di abitudini — *chi passa
la scena a chi* e *chi dice che cosa* — imparati contando su migliaia di
frasi già etichettate a mano. Etichettare una frase nuova è un ragionamento
da detective: sentite le battute «la porta cigola», qual è la sfilata di
attori dietro la tenda che le spiega meglio? Certezze non ce ne sono —
«porta» potrebbe dirla il NOME o il VERBO — ma puoi calcolare quale storia è
più probabile, ed è quella che scrivi.

`````

`````{tab} Superiore

Un HMM per il tagging ha come **stati nascosti** le etichette
$t_1, \dots, t_n$ e come **osservazioni** le parole $w_1, \dots, w_n$. Due
assunzioni lo definiscono: ogni etichetta dipende solo dalla precedente
(catena di Markov del primo ordine) e ogni parola dipende solo dalla
propria etichetta. La probabilità congiunta si fattorizza allora in

$$
P(t_1, \dots, t_n,\; w_1, \dots, w_n)
= \prod_{i=1}^{n} P(t_i \mid t_{i-1})\, P(w_i \mid t_i),
$$

dove $P(t_i \mid t_{i-1})$ sono le probabilità di **transizione** tra
etichette (con $t_0$ simbolo convenzionale di inizio frase) e
$P(w_i \mid t_i)$ le probabilità di **emissione** delle parole; entrambe si
stimano contando su un corpus annotato. Il tagging è la ricerca della
sequenza di stati più probabile date le parole,

$$
\hat{t}_{1:n} = \arg\max_{t_{1:n}} P(t_{1:n} \mid w_{1:n})
= \arg\max_{t_{1:n}} P(t_{1:n}, w_{1:n}),
$$

dove la seconda uguaglianza segue dalla regola di Bayes: il denominatore
$P(w_{1:n})$ non dipende dalle etichette. È un modello **generativo**:
descrive come etichette e parole vengono prodotte insieme, e la lettura
d'obbligo resta il tutorial di Rabiner {cite}`rabiner1989tutorial`.

`````

Il titolo del tutorial di Rabiner, non a caso, parla di *speech
recognition*: gli HMM sono la stessa macchina matematica che ritroveremo nel
capitolo sul riconoscimento vocale — sotto la sigla storica HMM-GMM — come
spina dorsale dell'ASR per trent'anni. Cambia solo il cast: là gli stati
nascosti sono i suoni elementari della lingua e le emissioni vettori
acustici; qui gli stati sono categorie grammaticali e le emissioni parole.
Quando in quel capitolo li vedrai nominare, la matematica sarà questa.

## Viterbi, o l'arte di non provarle tutte

Resta il problema pratico: *trovare* la sequenza di etichette più probabile.
Con 17 categorie e una frase di 20 parole, le sequenze possibili sono
$17^{20}$ — circa quattro milioni di miliardi di miliardi. Enumerarle è
fuori discussione. La salvezza è la struttura a catena del modello: per
decidere il meglio fino alla parola 12 basta sapere il meglio fino alla 11,
categoria per categoria. È la ricetta della **programmazione dinamica**, e
l'algoritmo che la applica porta il nome di Andrew Viterbi
{cite}`viterbi1967error` (nato Andrea a Bergamo nel 1935, emigrato bambino
negli Stati Uniti), che lo propose nel 1967 non per la grammatica ma
per decodificare segnali su canali rumorosi — lo stesso algoritmo ha poi
viaggiato nei decodificatori dei telefoni cellulari.

Facciamo i conti fino in fondo su un modello giocattolo: tre categorie
(`DET`, `NOME`, `VERBO`) e la frase «la porta cigola», dove «porta» ha la
stessa doppiezza dell'aggancio di questa sezione. Le probabilità di
partenza e di transizione:

| da ↓ verso → | `DET` | `NOME` | `VERBO` |
|---|---|---|---|
| inizio frase | 0,6 | 0,3 | 0,1 |
| `DET` | 0,1 | 0,7 | 0,2 |
| `NOME` | 0,2 | 0,3 | 0,5 |
| `VERBO` | 0,4 | 0,4 | 0,2 |

E le emissioni: `DET` dice «la» con probabilità 0,5; `NOME` dice «porta»
con 0,2; `VERBO` dice «porta» con 0,2 e «cigola» con 0,3; tutte le altre
combinazioni valgono 0. (Le righe non sommano a uno: la probabilità
restante va alle altre parole del vocabolario.) Nota il punto delicato: la
parola «porta», da sola, *non decide* — 0,2 contro 0,2. {numref}`fig-viterbi-traliccio`
mostra il **traliccio** (*trellis*): una colonna per parola, una casella
per categoria, e tutti i cammini che l'algoritmo valuta.

```{figure} ../figures/viterbi-traliccio.svg
:name: fig-viterbi-traliccio
:alt: Traliccio di Viterbi per la frase «la porta cigola» con tre stati DET, NOME e VERBO per colonna. Il cammino ottimo DET, NOME, VERBO è in terracotta con le probabilità parziali 0,30, 0,042 e 0,0063; il cammino alternativo che passa da VERBO su «porta» è in grigio; i cammini a probabilità zero sono tratteggiati.
:width: 100%

Il traliccio di Viterbi su «la porta cigola»: in ogni casella sopravvive
solo il migliore dei cammini che vi arrivano, e alla fine si risale
all'indietro lungo la strada in terracotta.
```

`````{tab} Elementare

Pensa a un navigatore che deve attraversare tre incroci (le tre parole),
scegliendo a ogni incrocio una corsia (la categoria). Il suo segreto: a
ogni incrocio, per ogni corsia, conserva **solo il modo migliore di
arrivarci** e butta via gli altri — se due strade sbucano nella stessa
corsia dello stesso incrocio, quella più lenta non potrà mai più
recuperare. Seguiamolo sui numeri della figura.

**«la»**: solo `DET` sa dire «la», quindi c'è una sola casella viva:
0,6 × 0,5 = **0,30**.

**«porta»**: due caselle possibili. Arrivare a `NOME` vale
0,30 × 0,7 × 0,2 = **0,042**; arrivare a `VERBO` vale
0,30 × 0,2 × 0,2 = **0,012**. La parola era in perfetto pareggio (0,2 e
0,2): a sbilanciare è la grammatica, quello 0,7 contro 0,2 — dopo un
articolo ci si aspetta un nome.

**«cigola»**: solo `VERBO` può dirla, ma ci si arriva da due strade: da
`NOME` vale 0,042 × 0,5 = 0,021, da `VERBO` vale 0,012 × 0,2 = 0,0024. Il
navigatore tiene la prima, moltiplica per l'emissione (× 0,3) e chiude a
**0,0063**. Ora risale i suoi appunti all'indietro: `VERBO` ← `NOME` ←
`DET`. Ecco l'etichettatura: *la*/articolo *porta*/nome *cigola*/verbo.

Qui i cammini erano una manciata; con 17 categorie e 20 parole sarebbero
miliardi di miliardi, ma gli *incroci* restano appena 17 × 20 = 340. Il
navigatore fa un pugno di moltiplicazioni per incrocio e trova comunque,
garantito, il percorso migliore in assoluto.

`````

`````{tab} Superiore

Definiamo $v_t(s)$ come la probabilità del miglior cammino che arriva allo
stato $s$ dopo aver generato le prime $t$ parole. L'algoritmo di Viterbi la
calcola per ricorrenza:

$$
v_1(s) = \pi_s \, P(w_1 \mid s),
\qquad
v_t(s) = \max_{s'} \big[\, v_{t-1}(s')\, P(s \mid s') \,\big]\, P(w_t \mid s),
$$

dove $\pi_s$ è la probabilità iniziale dello stato $s$ e $s'$ scorre su
tutti gli stati al passo precedente; un **retropuntatore**
$\psi_t(s) = \arg\max_{s'} v_{t-1}(s')\, P(s \mid s')$ memorizza da dove
proviene il massimo. Sul modello giocattolo la tabella dei massimi è:

| stato | «la» | «porta» | «cigola» |
|---|---|---|---|
| `DET` | **0,30** | 0 | 0 |
| `NOME` | 0 | **0,042** ← `DET` | 0 |
| `VERBO` | 0 | 0,012 ← `DET` | **0,0063** ← `NOME` |

Al termine si prende lo stato finale con $v_n$ massimo (qui `VERBO`, con
$0{,}0063$) e si segue $\psi$ a ritroso: `DET` → `NOME` → `VERBO`. Il
cammino alternativo completo `DET` → `VERBO` → `VERBO` vale
$0{,}012 \times 0{,}2 \times 0{,}3 = 0{,}00072$: quasi dieci volte meno.
Il costo è $O(n\,T^2)$ — per ogni parola, per ogni stato, un massimo su $T$
predecessori — contro gli $O(T^n)$ cammini della forza bruta: con $T = 17$
e $n = 20$, poche migliaia di operazioni al posto di $10^{24}$, e con la
garanzia dell'ottimo globale — a differenza della *beam search* della
sezione precedente, che è un'euristica. In pratica si lavora con i
logaritmi, sommando invece di moltiplicare, per evitare l'underflow.

`````

Una riga di storia successiva: agli HMM, generativi, sono succeduti i
**Conditional Random Field** (CRF) {cite}`lafferty2001conditional`, la
variante *discriminativa* che modella direttamente $P(t_{1:n} \mid w_{1:n})$
e può usare caratteristiche arbitrarie delle parole (maiuscole, suffissi,
trattini) — per oltre un decennio lo stato dell'arte del NER, con la
decodifica affidata sempre a Viterbi.

## La via neurale: una BiLSTM per etichettare

E le reti ricorrenti? Nella sezione sulla traduzione abbiamo stabilito una
regola: la lettura **bidirezionale** vale solo per *capire* un testo che
esiste già tutto intero, non per generarlo. L'etichettatura è il caso ideale
— la frase è lì, completa, e per decidere l'etichetta di «porta» servono
tanto le parole prima quanto quelle dopo («la porta **cigola**» contro «la
porta **a scuola**»). Un tagger neurale minimo è quindi: embedding, LSTM
bidirezionale, e uno strato lineare che produce un punteggio per etichetta
*per ogni token*.

```python
import torch
from torch import nn

class TaggerBiLSTM(nn.Module):
    def __init__(self, vocab=10000, num_tag=17, dim=64, hidden=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim)   # parola -> vettore
        self.lstm = nn.LSTM(dim, hidden, batch_first=True,
                            bidirectional=True)     # legge nei due sensi
        self.out = nn.Linear(2 * hidden, num_tag)   # un logit per etichetta

    def forward(self, x):          # x: (batch, lunghezza), indici di parole
        e = self.embedding(x)      # (batch, lunghezza, dim)
        h, _ = self.lstm(e)        # (batch, lunghezza, 2*hidden)
        return self.out(h)         # logit per OGNI parola, non solo l'ultima
```

Confrontalo con il classificatore di sentiment della sezione sui modelli di
sequenza: là si teneva solo l'ultimo stato (`h[:, -1]`), un'etichetta per
frase; qui si tengono tutti, un'etichetta per parola. La loss è la solita
cross-entropia, applicata token per token:

```{code-block} python
:class: pt-non-eseguibile

modello = TaggerBiLSTM()
perdita = nn.CrossEntropyLoss(ignore_index=-100)  # -100 = padding da ignorare

# frasi: (batch, lunghezza), indici di parole; tag: stessa forma, -100 sul padding
logits = modello(frasi)                    # (batch, lunghezza, 17)
loss = perdita(logits.reshape(-1, 17),     # una riga per token
               tag.reshape(-1))            # un'etichetta per token
loss.backward()                            # poi optimizer.step(), come sempre
```

Nei sistemi di punta questa ricetta si arricchisce spesso di uno strato CRF
finale, che rimette in gioco le transizioni tra etichette (una `I-PER`
subito dopo una `O` non ha senso, e il CRF lo sa). Oggi però lo standard del
NER è un'altra strada: prendere un modello pre-addestrato come BERT
{cite}`devlin2019bert`, aggiungergli la stessa testa lineare per token e
fare *fine-tuning* — ne parleremo nel capitolo sui Transformer, dove la
bidirezionalità qui costruita con due LSTM diventerà una proprietà nativa
dell'attenzione.

## Misurare bene: token o entità?

Un'ultima questione, meno contabile di quanto sembri: come si dà il voto a
un etichettatore? La risposta giusta è diversa per i due compiti, e la
differenza insegna qualcosa.

`````{tab} Elementare

Per le parti del discorso il voto naturale funziona: quante parole hanno
ricevuto l'etichetta giusta, su cento. Ma attenzione alle percentuali
gonfiate: siccome tante parole sono facili («il» è quasi sempre articolo),
anche un sistema mediocre parte da voti alti — è il 97% che è difficile,
non il 90%.

Per le entità quel voto diventa una trappola. Prendi un testo di 100 parole
che contiene una sola entità, «Enrico Fermi». Un sistema pigro che non
evidenzia *niente* azzecca 98 parole su 100: 98%, e non ha trovato nulla!
E un sistema che evidenzia solo «Enrico», lasciando fuori «Fermi», ha
prodotto un'entità sbagliata: mezza persona non serve a nessuno. Per questo
il NER si giudica a evidenziature intere — vale solo il segmento completo,
del colore giusto — e con due domande: di quello che hai evidenziato,
quanto era giusto? E di quello che andava evidenziato, quanto ne hai
trovato? Sono la precisione e il richiamo che abbiamo incontrato nel
capitolo sul machine learning, riuniti nel loro voto unico, l'F1.

`````

`````{tab} Superiore

Per il POS tagging la metrica è l'**accuratezza per token**,
$\text{acc} = \frac{\#\{i : \hat{t}_i = t_i\}}{n}$. Va letta contro una
base di confronto onesta: il baseline «assegna a ogni parola la sua
etichetta più frequente nel corpus» supera già il 92% sull'inglese
giornalistico, quindi lo spazio reale di miglioramento — dal 92% al 97% e
oltre — sta tutto nelle occorrenze ambigue, le più difficili.

Per il NER si usano precisione, richiamo e F1 **a livello di entità**, con
il criterio dell'*exact match* reso standard dalle campagne di valutazione
CoNLL dei primi anni Duemila: un'entità predetta conta come corretta solo
se coincidono sia i confini del segmento sia il tipo. Dette $C$ le entità
corrette, $\hat{E}$ quelle predette ed $E$ quelle di riferimento:

$$
P = \frac{|C|}{|\hat{E}|},
\qquad
R = \frac{|C|}{|E|},
\qquad
F_1 = \frac{2PR}{P + R},
$$

dove $P$ (precisione) misura quanto ci si può fidare di ciò che il sistema
estrae e $R$ (richiamo) quanta parte del dovuto viene trovata. La severità
dell'exact match è motivata dall'uso a valle: un'entità dai confini
sbagliati inquina qualunque base di conoscenza la riceva. L'accuratezza per
token, dominata dall'etichetta `O`, qui non discrimina nulla: il sistema
che predice sempre `O` ne uscirebbe con punteggi altissimi e utilità zero.

`````

Chiudiamo dove avevamo aperto. Davanti a «La vecchia porta la sbarra» un
tagger sceglie la lettura più probabile secondo la sua esperienza — quasi
certamente la signora con la sbarra, perché le statistiche della lingua
pendono da quella parte. Ma sapere che «porta» è un verbo non dice ancora
*chi fa che cosa a chi*: per questo bisogna salire di un piano, dalle
etichette alla struttura della frase. È l'analisi sintattica, tema della
prossima sezione — e i suoi mattoni sono esattamente le etichette POS che
abbiamo imparato a mettere.

```{admonition} Da ricordare
:class: important
- Il **POS tagging** assegna a ogni parola la sua categoria grammaticale:
  lo standard è il tagset **universale** a 17 categorie di Universal
  Dependencies. Le parole ambigue («porta», «ancora») sono poche nel
  vocabolario ma frequentissime nei testi: decide il contesto.
- Il **NER** trova persone, luoghi, organizzazioni e date; lo **schema
  BIO** (`B-X`, `I-X`, `O`) lo trasforma in un'etichetta per token e tiene
  distinte le entità adiacenti.
- Lo **HMM** è un modello generativo con stati nascosti (le etichette) e
  osservazioni (le parole): $P(t,w) = \prod_i P(t_i \mid t_{i-1}) P(w_i \mid t_i)$;
  transizioni ed emissioni si contano su un corpus annotato. La stessa
  macchina, con i suoni al posto delle etichette, regge l'ASR storico.
- L'algoritmo di **Viterbi** trova la sequenza di stati ottima con la
  programmazione dinamica sul traliccio: $O(n\,T^2)$ invece di $O(T^n)$,
  tenendo in ogni casella solo il miglior cammino in arrivo. I **CRF** sono
  la variante discriminativa.
- Il tagger neurale è una **BiLSTM** con testa lineare per token e
  cross-entropia per token: la bidirezionalità è legittima perché il testo
  è tutto disponibile. Oggi il NER di punta è fine-tuning di BERT.
- Valutazione: **accuratezza per token** per il POS (con baseline già oltre
  il 92%), **F1 a livello di entità** con exact match per il NER — perché
  mezza entità è un'entità sbagliata.
```
