# I modelli di riconoscimento

Quando pronunci la parola «casa», il microfono del telefono non registra
quattro lettere: registra circa sedicimila numeri al secondo, un fiume di
campioni che descrivono come vibra l'aria. Il compito del riconoscimento
vocale automatico (*Automatic Speech Recognition*, ASR) è tradurre quel fiume
in una manciata di caratteri. Sembra un problema di traduzione come un altro,
ma nasconde una difficoltà tutta sua, che ha condizionato per decenni il modo
in cui si costruiscono questi modelli.

## Il problema dell'allineamento

Prima di dare in pasto l'audio a una rete, lo trasformiamo in uno
**spettrogramma**: tagliamo il segnale in finestrelle di circa 25 millisecondi
(i *frame*) e per ciascuna misuriamo quanta energia c'è a ogni frequenza. Un
secondo di parlato diventa così un centinaio di frame. La trascrizione, invece,
è lunga poche decine di caratteri. Due sequenze di lunghezza molto diversa — e
nessuno ci dice quale frame corrisponde a quale lettera.

`````{tab} Elementare
Immagina di dover sottotitolare un video a orecchio, senza conoscere i tempi.
Chi parla lento, chi veloce; una vocale tenuta a lungo — «caaasa» — occupa
molti fotogrammi ma resta una sola lettera; tra una parola e l'altra ci sono
pause e respiri che non vanno scritti. Sai *cosa* è stato detto, ma non
*quando* comincia e finisce ogni suono. Questo è l'allineamento: appaiare i
tanti pezzetti di audio ai pochi caratteri del testo.
`````

`````{tab} Superiore
Abbiamo un input $X = (x_1, \dots, x_T)$ di $T$ frame e un target
$\mathbf{y} = (y_1, \dots, y_U)$ di $U$ token (caratteri o sotto-parole), con
$T \gg U$. L'allineamento è **monotono** (l'audio scorre in avanti come il
testo) ma **sconosciuto**: non abbiamo etichette frame per frame. Segmentare a
mano milioni di ore per dire «da qui a qui c'è una a» è impraticabile. Serve un
modello che impari l'allineamento *da solo*, dalla sola coppia
(audio, trascrizione).
`````

```{figure} ../figures/ctc-allineamento.svg
:name: fig-ctc-allineamento
:alt: Sette frame audio, ciascuno etichettato con un simbolo P A A L vuoto L A; una freccia collassa i ripetuti e rimuove i simboli vuoti, producendo la parola PALLA.
:width: 85%

Il meccanismo della CTC. La rete assegna un simbolo a ogni frame (anche il
simbolo «vuoto» ∅); la regola di collasso unisce i ripetuti consecutivi e poi
toglie i vuoti. Il ∅ tra le due «L» serve proprio a preservare la doppia.
```

## CTC: imparare ad allineare da soli

La svolta arriva nel 2006 con la **Connectionist Temporal Classification** di
Alex Graves e colleghi. L'idea è aggiungere all'alfabeto un simbolo speciale, il
«vuoto» (*blank*, $\varnothing$), che significa «qui non produco nessun
carattere». La rete emette, per **ogni** frame, una distribuzione su tutti i
simboli più il vuoto. Poi una regola di collasso $\mathcal{B}$ ripulisce la
sequenza: prima unisce i caratteri uguali consecutivi, poi elimina i vuoti
({numref}`fig-ctc-allineamento`).

`````{tab} Elementare
Molti modi di etichettare i frame danno la stessa parola. Per «PALLA» va bene
`P A A L ∅ L A`, ma anche `P P A L ∅ L L A`: entrambi, dopo aver unito i doppioni
e tolto i vuoti, diventano `PALLA`. La CTC non sceglie *un* allineamento
giusto: li considera tutti insieme e premia la rete se la loro somma punta
verso la trascrizione corretta. Nota il trucco del vuoto: senza il ∅ in mezzo,
le due «L» si fonderebbero in una sola.
`````

`````{tab} Superiore
La probabilità di una trascrizione $\mathbf{y}$ è la somma su tutti i percorsi
frame-level $\pi$ che, collassati, la producono:

$$
p(\mathbf{y} \mid X) = \sum_{\pi \,\in\, \mathcal{B}^{-1}(\mathbf{y})}
\prod_{t=1}^{T} p_t(\pi_t \mid X),
$$

dove $\pi = (\pi_1, \dots, \pi_T)$ è un allineamento a livello di frame,
$p_t(\pi_t \mid X)$ è la probabilità che la rete assegna al simbolo $\pi_t$ al
frame $t$, e $\mathcal{B}$ è la funzione di collasso. La somma ha un numero
esponenziale di termini, ma si calcola in tempo lineare con un algoritmo di
programmazione dinamica *forward-backward*. Si addestra minimizzando
$\mathcal{L} = -\log p(\mathbf{y} \mid X)$. Il limite noto: la CTC assume che le
predizioni ai vari frame siano **condizionatamente indipendenti** dato $X$,
quindi non modella bene le dipendenze fra i caratteri in uscita.
`````

## Ascoltare e attendere: i modelli con attenzione

Un'alternativa evita del tutto il vuoto. I modelli **sequenza-a-sequenza con
attenzione** hanno un *encoder* che riassume tutto l'audio e un *decoder* che
genera i caratteri uno alla volta, ciascuno tenendo conto di quelli già
scritti (in gergo, in modo *autoregressivo*). A ogni passo il
decoder usa l'**attenzione** per decidere su quali frame concentrarsi: è un
allineamento «morbido», appreso, non deciso a priori. L'architettura di
riferimento è *Listen, Attend and Spell* {cite}`chan2016listen`.

`````{tab} Elementare
Pensa a un interprete: prima ascolta l'intera frase, poi la ridice parola per
parola. Mentre pronuncia ogni parola, la sua attenzione torna al punto giusto
di ciò che ha sentito. Il modello fa lo stesso: genera un carattere, si
«riguarda» la porzione di audio più rilevante, genera il prossimo.
`````

`````{tab} Superiore
Al passo $i$ il decoder costruisce un vettore di contesto come media pesata
degli stati dell'encoder $\mathbf{h}_j$:

$$
\mathbf{c}_i = \sum_{j=1}^{T} \alpha_{ij}\,\mathbf{h}_j,
\qquad \sum_j \alpha_{ij} = 1,
$$

dove i pesi di attenzione $\alpha_{ij}$ dicono quanto il frame $j$ conta per
produrre il token $i$-esimo. A differenza della CTC, il decoder condiziona ogni
token su quelli già emessi: modella cioè le dipendenze del testo in uscita, al
prezzo di una generazione sequenziale più lenta.
`````

## Whisper e i Transformer end-to-end

Nel settembre 2022 OpenAI rilascia **Whisper**: un unico Transformer
encoder-decoder che riceve lo spettrogramma log-mel e produce direttamente il
testo. La sua forza non è tanto l'architettura quanto i dati: 680.000 ore di
audio raccolte dal web con etichettatura debole — trascrizioni prese così
com'erano, senza revisione umana — in oltre novanta lingue. Con lo
stesso modello Whisper trascrive, traduce verso l'inglese e riconosce la lingua,
guidato da istruzioni speciali (*token*) inserite nel decoder. È robusto ad accenti e rumore e ha reso
obsoleta la vecchia pipeline a stadi (estrazione feature, modello acustico,
dizionario di pronuncia, modello di linguaggio). I limiti restano onesti: su
silenzi lunghi può «allucinare» testo o ripetersi in loop.

## Il modello di linguaggio, il correttore silenzioso

L'evidenza acustica, da sola, è ambigua. In italiano «l'ago» e «lago», «l'una»
e «luna» suonano identici; è il contesto a decidere. Qui entra il **modello di
linguaggio** (LM), che assegna una probabilità alle sequenze di parole plausibili
e sposta la trascrizione verso ciò che «suona» come italiano corretto. Nei
sistemi classici i due giudizi — quello dell'orecchio e quello del linguaggio —
si sommano in un punteggio unico, con un peso che regola quanto contare il
secondo (*shallow fusion*); in alternativa il LM si usa per riordinare le
prime $n$ ipotesi di trascrizione. I modelli end-to-end come Whisper
imparano un LM *implicito* dal loro stesso addestramento; un LM esterno resta
comunque utile per termini rari o di dominio: nomi propri, sigle, gergo medico.

## Misurare gli errori: il Word Error Rate

Come diciamo che una trascrizione è «buona»? La metrica standard è il **Word
Error Rate** (WER): la distanza di edit fra la trascrizione prodotta e quella di
riferimento, contata a livello di parola.

$$
\text{WER} = \frac{S + D + I}{N},
$$

dove $S$ è il numero di **sostituzioni**, $D$ le **cancellazioni**, $I$ le
**inserzioni** e $N$ il numero di parole nel riferimento. Un WER di $0$ è la
trascrizione perfetta; può superare $1$ se il modello inserisce più parole di
quante ce ne siano. Calcolarlo è un classico problema di distanza di
Levenshtein:

```python
import numpy as np

def wer(rif, ip):
    r, h = rif.split(), ip.split()
    # matrice di distanza di edit riempita per programmazione dinamica
    D = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    D[:, 0] = np.arange(len(r) + 1)   # cancellazioni pure
    D[0, :] = np.arange(len(h) + 1)   # inserzioni pure
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            costo = 0 if r[i - 1] == h[j - 1] else 1
            D[i, j] = min(D[i - 1, j] + 1,          # cancellazione
                          D[i, j - 1] + 1,          # inserzione
                          D[i - 1, j - 1] + costo)  # sostituzione (o parola uguale)
    return D[len(r), len(h)] / len(r)

wer("il gatto nero salta sul muro",
    "il gatto nemo salta muro")   # -> 0.333  (1 sostituzione + 1 cancellazione su 6 parole)
```

Il WER è comodo ma grezzo: pesa allo stesso modo un errore grave e uno banale,
e penalizza le lingue ricche di composti. Per questo, accanto a esso, si riporta
spesso il *Character Error Rate* (CER), che conta gli stessi errori a livello di
carattere. Nessuna metrica, però, cattura del tutto ciò che conta davvero: se la
frase trascritta, letta da un essere umano, significa ancora la cosa giusta.

```{admonition} Da ricordare
:class: important
- Audio e testo hanno **lunghezze diverse** e l'allineamento non è dato: è il
  problema centrale dell'ASR.
- La **CTC** lo risolve con il simbolo «vuoto» e sommando tutti gli
  allineamenti possibili; i modelli **con attenzione** lo imparano in modo
  morbido, un token alla volta.
- I Transformer end-to-end come **Whisper** {cite}`radford2022robust`
  uniscono tutto in un solo modello multilingue.
- Il **modello di linguaggio** disambigua gli omofoni; il **WER** misura gli
  errori come distanza di edit fra parole.
```
