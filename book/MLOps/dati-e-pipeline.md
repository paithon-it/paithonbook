# I dati contano quanto il programma

Apri il registro delle modifiche di un sistema che riconosce le frodi con la
carta di credito, mentre è in funzione davanti a clienti veri. Il registro è
quello di `git`, il programma con cui il mondo del software tiene la propria
cronologia: ogni modifica ci finisce dentro con la sua data e il suo perché (si
chiama *commit*). E il codice del modello è quasi fermo: qualche riga nuova al
mese, una libreria aggiornata, una manopola ritoccata. Poi
guarda i dati che quel codice ingoia: milioni di transazioni ogni giorno, mai
due uguali, con nuovi negozi, nuovi importi, nuove truffe che ieri non
esistevano. Il codice è la strada; i dati sono il traffico che ci passa sopra.
Nel software tradizionale la cosa che cambia (e che quindi si conserva versione
per versione, si controlla, si sorveglia) è la strada. Nel machine learning la
strada sta quasi ferma, e a muoversi sotto i piedi è il traffico.

Da questa asimmetria nasce un'idea che negli ultimi anni ha un nome:
**data-centric AI**. La provocazione, resa popolare da Andrew Ng intorno al
2021, è semplice. Per un decennio si sono limate le architetture per rubare un
decimale di accuratezza a un *benchmark*, cioè a una prova standard su cui i
modelli si confrontano, come un compito in classe uguale per tutti. Ma nei
sistemi reali il guadagno più grande si ottiene quasi sempre migliorando i
*dati*: etichette più coerenti, esempi più rappresentativi, meno rumore.

Se è così, i dati non possono restare un allegato del codice: vanno trattati
come **cittadini di prima classe**, versionati, testati e sorvegliati con la
stessa disciplina. La sezione precedente ha stabilito che riprodurre un
modello richiede tre artefatti: codice, dati, modello. Questa entra nel più
grande e trascurato dei tre, e nel sistema di tubature che lo trasporta
{cite}`huyen2022designing`.

## Versionare i dati

Se non sai *quali* dati hanno prodotto un modello, non puoi riprodurlo, non
puoi capire perché una predizione è quella che è, e non puoi tornare indietro
quando un nuovo addestramento peggiora le cose. Del codice `git` conserva ogni
versione da decenni, ma sui dati, per il motivo già visto nella pagina
d'apertura, non funziona: una raccolta di dati (un **dataset**) è enorme e non è
fatta di righe da confrontare, e metterci dentro dieci gigabyte di immagini
gonfia il progetto fino a renderlo inservibile. Per i dati serve quindi qualcosa
di diverso, che faccia lo stesso mestiere.

`````{tab} Elementare

Hai presente la cronologia di un documento condiviso: un Google Doc, una
pagina di Wikipedia? Puoi tornare a com'era martedì scorso, vedere chi ha
cambiato cosa, recuperare un paragrafo cancellato per sbaglio. Versionare i
dati è la stessa idea, applicata a una tabella o a una cartella di immagini: a
ogni «versione» del dataset resta appeso un cartellino con un codice, e
citando quel codice chiunque ritrova *esattamente* quei dati, non una loro
versione simile.

Il trucco per farlo senza duplicare montagne di file è un'impronta digitale.
Immagina di passare l'intero dataset in un tritatutto che ne ricava un codice
corto (poche decine di caratteri) con una proprietà magica: se cambi anche un
solo pixel, il codice cambia del tutto; se il dataset è identico, il codice è
identico. Così, invece di conservare cento copie dei dati dentro il progetto,
conservi solo i cartellini, e i dati veri stanno una volta sola in un
magazzino a parte. Il progetto dice «mi servono i dati con il cartellino
`a3f8…`», e il magazzino li consegna.

`````

`````{tab} Superiore

L'impronta è un **hash crittografico** del contenuto (per esempio SHA-256):
una funzione che mappa una sequenza arbitraria di byte in una stringa di
lunghezza fissa, deterministica e sensibile a ogni bit. Su questa idea si
costruisce il **content-addressable storage**: l'indirizzo di un dato *è*
l'hash del suo contenuto, non un percorso né un nome scelto a mano. Ne
discendono tre proprietà preziose. **Immutabilità**: un dato non si modifica
«sul posto», si scrive una nuova versione con un nuovo indirizzo; le vecchie
restano raggiungibili, e un esperimento passato non cambia sotto i piedi.
**Deduplicazione**: file identici hanno lo stesso hash e occupano spazio una
volta sola. **Integrità**: ricalcolare l'hash verifica che il dato non si sia
corrotto in transito.

Nel repository si versiona allora solo un *puntatore* (un piccolo file di
testo che contiene l'hash e l'indirizzo del magazzino remoto), mentre
l'artefatto vero vive in un *object store*, cioè un archivio di rete che
conserva blocchi di byte e li restituisce a chi ne conosce la chiave. Esistono
strumenti che industrializzano questo
schema, ma non aggiungono niente al concetto: è lo stesso meccanismo con cui
`git` identifica i propri oggetti. Sopra i puntatori si costruisce infine il
**lineage** (la *provenienza*): il grafo che lega ogni modello alla versione
esatta dei dati e del codice che lo hanno generato. È ciò che permette, mesi
dopo, di rispondere alla domanda più temuta in un audit («da dove viene questa
predizione?») risalendo la catena fino al singolo dato grezzo.

`````

## Le pipeline di dati

Un dataset pronto per l'addestramento non nasce così: è il prodotto finale di
una catena di trasformazioni. Si **estrae** il dato grezzo, cioè come arriva
dal mondo, da una o più sorgenti (un archivio, un flusso di eventi, dei file);
si **pulisce**, buttando via i doppioni e sistemando le caselle vuote e i
formati incoerenti; si costruiscono le **feature**, cioè le poche grandezze che
si mettono davanti al modello al posto del dato grezzo («la spesa media
dell'ultimo mese», «quante volte ha comprato di notte»); e solo alla fine si
**addestra**. È la *pipeline* annunciata nella pagina d'apertura: il dato entra
da un capo, attraversa una stazione dopo l'altra e ne esce pronto. Ognuno di
questi passaggi si scrive con gli attrezzi per maneggiare tabelle già visti nel
{doc}`capitolo su Python </Python/overview>`.

Il salto di qualità, però, non è tecnico: è organizzativo, e chiede due cose
alla catena. Che sia **riproducibile**, cioè che rilanciandola sugli stessi
dati grezzi si riottenga lo stesso identico dataset. E che sia **orchestrata**,
cioè che l'ordine dei passaggi sia scritto una volta in un file, e sia un
programma a farli partire in quell'ordine, invece di una persona che li lancia
a mano in un notebook e ogni tanto ne salta uno.

```{figure} ../figures/feature-engineering.svg
:name: fig-feature-engineering
:alt: "Catena in tre blocchi: i dati grezzi entrano nel blocco di feature engineering, che li trasforma nelle variabili con cui il modello lavora, e solo queste ultime raggiungono il modello. Il modello non vede mai i dati grezzi."
:width: 90%

Il modello non vede i dati: vede le feature. Tutto ciò che si decide nel
blocco di mezzo è il mondo dentro cui il modello dovrà cavarsela.
```

La freccia a senso unico di {numref}`fig-feature-engineering` dice una cosa
sola: quello che il modello sa del mondo passa tutto per il blocco di mezzo, e
niente lo scavalca. Ne segue che quel blocco va conservato versione per
versione esattamente come il programma. Se cambia il modo di costruire una
feature, il modello addestrato prima e quello addestrato dopo non stanno più
guardando la stessa cosa, e confrontare i loro voti è come confrontare i tempi
di due corse su piste di lunghezza diversa.

`````{tab} Elementare

Il latte che arriva in bottiglia ha attraversato una filiera: la cisterna lo
raccoglie dalle stalle, il laboratorio ne analizza un campione, la
pastorizzazione lo tiene alla temperatura giusta per il tempo giusto, e solo
alla fine si imbottiglia. Nessuno di quei passaggi si salta, e l'ordine non è
un dettaglio: se il campione si analizzasse in fondo, un carico guasto lo si
scoprirebbe quando è già nelle bottiglie di tutti, mentre analizzato all'inizio
quel carico si scarta e non entra. Una pipeline di dati è questo: passaggi
collegati, ognuno con un compito e con il suo controllo, che trasformano il
dato grezzo in dato pronto senza che nessuno intervenga a mano ogni volta.

Il pericolo ha un nome pittoresco: la **pipeline jungle**, la «giungla di
tubature». Nasce così: qualcuno aggiunge di fretta uno script per sistemare un
caso particolare, poi un altro sopra il primo per rattoppare un nuovo problema,
e mesi dopo nessuno capisce più quale tubo alimenta quale, né cosa succede se ne
tocchi uno. La cura non è un attrezzo magico ma una disciplina: passaggi
piccoli, dichiarati, ognuno ripetibile da solo, e nessuna trasformazione «a
mano» che non lasci traccia.

`````

`````{tab} Superiore

Formalmente una pipeline è un **DAG** (*directed acyclic graph*): i nodi sono
trasformazioni, gli archi le dipendenze dato-verso-dato, e l'assenza di cicli
garantisce un ordine di esecuzione ben definito. Due proprietà la rendono
governabile. L’**idempotenza**: rieseguire uno stadio sugli stessi input
produce lo stesso output, senza effetti collaterali accumulati (condizione per
poter ripartire da metà catena dopo un errore). E la **materializzazione
versionata** degli stadi intermedi, così che un cambiamento a valle non
obblighi a ricalcolare tutto da capo. Il programma che tiene insieme il tutto
si chiama **orchestratore**: decide in che ordine far girare gli stadi, quali
possono andare in parallelo e cosa ritentare quando uno fallisce. Ma, come per
il versionamento, lo strumento è secondario rispetto al principio.
Automatizzare l'intera catena (da dato grezzo a modello valutato)
con un comando solo è il cuore della *Continuous Delivery for Machine
Learning* {cite}`sato2019continuous`: finché un pezzo della pipeline resta un
rito manuale, l'intero sistema non è né riproducibile né rilasciabile in modo
affidabile.

`````

### In che formato stanno i dati, e perché conta

C'è una decisione che si prende all'inizio, che sembra tecnica e non lo è: come
i dati stanno scritti su disco fra uno stadio e il successivo. Quasi tutti,
senza pensarci, scelgono il **CSV**, che è il modo più semplice di scrivere una
tabella in un file di testo: una riga del file per ogni riga della tabella, e
dentro ogni riga i valori separati da una virgola. È quello che esce da un
foglio di calcolo quando gli si chiede di esportare. Ed è quasi sempre la
scelta sbagliata.

`````{tab} Elementare

Ottanta informazioni per ogni cliente, un milione di clienti: un archivio così
si conserva in due modi. **Per riga**, cioè una scheda per cliente con tutte le
sue ottanta voci in fila: è il CSV, ed è comodo se ti serve tutta la scheda di
un cliente. Oppure **per colonna**, cioè un elenco di tutte le età, poi un
elenco di tutte le città, e così via.

Quale conviene dipende da cosa fai. E quello che si fa per addestrare un
modello è sempre lo stesso: leggere **tre colonne su ottanta**, per tutti. Con
l'archivio per riga devi attraversare l'intero milione di schede e scartare il
$96\%$ di ciò che leggi (77 voci buttate ogni 80). Non è pigrizia di chi ha
scritto il programma: un disco non consegna un valore alla volta, consegna
blocchi interi, e le tre voci che ti servono sono sparse dentro un milione di
schede, una qui e una là. Prenderle senza prendere anche il resto è
impossibile. Con l'archivio per colonna, invece, i tre elenchi che ti servono
stanno già tutti insieme: li prendi e il resto non lo tocchi nemmeno.

C'è un secondo guadagno, meno ovvio e spesso più grosso: **valori simili stanno
vicini**. In una colonna di città ci sono migliaia di «Milano» di fila, in una
di date ci sono numeri che crescono di poco alla volta. Roba del genere si
comprime benissimo, mentre in una scheda per riga ogni valore è circondato da
valori di natura diversa e la compressione ha poco da mordere.

Il formato per colonna più usato si chiama **Parquet**, e in più tiene i tipi
delle colonne (che il CSV non ha: per lui è tutto testo, ed è il motivo per cui
un codice postale che comincia per zero si trasforma in un numero e perde lo
zero).

Poi c'è **Arrow**, che risolve un problema diverso: non come i dati stanno sul
disco, ma come stanno **in memoria**. Sono due posti diversi dentro un
computer: il disco è l'armadio, dove le cose restano anche a macchina spenta;
la memoria è il tavolo su cui le tiri fuori per lavorarci, molto più veloce e
molto più piccolo.

Il guaio è che ogni programma, sul suo tavolo, dispone i dati a modo proprio.
Quando due programmi si passano una tabella, il primo deve quindi riscriverla
nella forma che il secondo capisce, e su una tabella grande quella traduzione
costa più del lavoro vero. Arrow è l'accordo che toglie di mezzo il problema:
se tutti e due dispongono i dati alla stessa maniera, non c'è niente da
riscrivere. Il secondo programma guarda direttamente lo stesso pezzo di tavolo
del primo, e il passaggio non costa nulla. È il motivo per cui questo accordo,
che nessuno nomina mai, sta sotto strumenti che sembrano non avere niente in
comune, compresi quelli con cui si maneggiano le tabelle in Python.

`````

`````{tab} Superiore

Un formato **orientato alle righe** (CSV, JSON Lines, Avro) memorizza i record
uno dopo l'altro; uno **orientato alle colonne** (**Parquet**, ORC) memorizza
insieme tutti i valori di una stessa colonna. La differenza produce tre effetti
che contano tutti in un carico di lavoro analitico.

**Projection pushdown**: leggere $k$ colonne su $d$ costa in proporzione a $k$
e non a $d$, perché le altre non vengono nemmeno toccate su disco. Nei carichi
di ML, dove si leggono poche colonne di tabelle larghe, è la voce dominante.

**Compressione**: dentro una colonna i valori sono omogenei per tipo e spesso
per contenuto, il che abilita codifiche specializzate (dizionario per le
categorie a bassa cardinalità, run-length per i valori ripetuti, delta per i
timestamp) prima ancora della compressione generica. Rispetto al CSV
equivalente il guadagno è di **qualche volta**, e a decidere quante è proprio
la prima di quelle codifiche. I numeri che seguono sono misurati qui, su
tabelle da duecentomila righe, confrontando il `.csv` e il `.parquet` scritti
da Pandas con le impostazioni di serie. Sei colonne di categorie con sei valori
distinti (nomi di città) stanno in un file **diciotto volte** più piccolo,
perché il dizionario sostituisce ogni stringa con un indice, e quante volte lo
decide la lunghezza delle stringhe. Sei colonne di numeri casuali con la
virgola scendono a **poco più di due volte**, perché lì non c'è niente da
riconoscere, e una tabella mista come quelle su cui si addestra di solito sta
fra il due e il tre a seconda di quante colonne siano categoriche.

Le colonne ordinate, che l'intuizione metterebbe in alto, non ci vanno, e la
ragione è istruttiva: la codifica che le comprimerebbe davvero (memorizzare le
differenze fra un valore e il precedente, la *delta encoding*) **non è quella
che la libreria sceglie da sola**. Su una colonna di istanti che crescono di
pochi secondi alla volta, il default si ferma a **due volte**, perché il
dizionario, su valori quasi tutti diversi, non ha niente da riusare; chiedendo
esplicitamente la delta si arriva a **quasi quaranta**. È il caso da tenere a
mente ogni volta che si dichiara che cosa fa uno strumento «di serie»: qui
l'impostazione di serie lascia sul tavolo un fattore venti.

**Predicate pushdown**: Parquet memorizza per ogni gruppo di righe le
statistiche di ciascuna colonna (minimo, massimo, conteggio dei nulli), quindi
un filtro `data > 2026-01-01` può **saltare interi blocchi** senza
decomprimerli.

A questo si aggiunge una cosa che il CSV strutturalmente non ha: uno **schema**
con i tipi. Un CSV è testo, e ogni lettore riscopre i tipi per euristica, il
che è la sorgente di una classe intera di bug silenziosi (l'identificativo con
gli zeri iniziali letto come intero, la data interpretata secondo la
convenzione locale, il campo vuoto che diventa `NaN` oppure la stringa
`"NA"` a seconda del lettore).

**Apache Arrow** risolve un problema ortogonale: è una specifica di
rappresentazione **in memoria**, colonnare, indipendente dal linguaggio. Il suo
valore è l'eliminazione della **serializzazione** ai confini: due processi, o
due librerie in linguaggi diversi, che parlano Arrow si scambiano una tabella
senza copiarla né convertirla. È la ragione per cui lo stesso formato compare
sotto motori che non si somigliano affatto, ed è anche ciò che sta sotto il
tipo stringa di Pandas. Vale la pena provarlo, perché è cambiato di recente e
in silenzio: dalla versione 3 le colonne di testo hanno un tipo dedicato,
appoggiato ad Arrow quando PyArrow è installato, e quella differenza (con il
vecchio `object` di NumPy era sostanziale) non è più un'opzione da attivare, è
il comportamento normale della libreria.

La regola pratica, sintetica: **CSV per scambiare con un umano, Parquet per
tutto il resto**; e se una tabella attraversa un confine di processo o di
linguaggio, Arrow.

`````

## Il feature store

C'è un punto della pipeline che merita un discorso a sé: le **feature**. Una
stessa feature («spesa media dell'utente negli ultimi 30 giorni», «numero di
transazioni nell'ultima ora») serve a più modelli. E soprattutto va calcolata
in *due momenti diversi*: una volta mentre il modello impara, su montagne di
dati vecchi, e un'altra mentre il modello risponde, su un singolo caso appena
arrivato. Se i due calcoli divergono anche di poco, il modello in funzione
riceve qualcosa di diverso da ciò su cui ha imparato.

Il rimedio è un magazzino unico delle feature, dove ciascuna è definita una
volta sola e da cui la prendono tutti e due, chi addestra e chi risponde. Si
chiama **feature store** {cite}`huyen2022designing`.

`````{tab} Elementare

Immagina una grande cucina con venti cuochi. Se ognuno si prepara il soffritto a
modo suo, ogni piatto esce leggermente diverso e nessuno sa più perché. La
soluzione è una **dispensa comune**: il soffritto lo prepara un reparto solo,
sempre con la stessa ricetta, e tutti i cuochi lo prendono già pronto da lì. Il
piatto di oggi è identico a quello di ieri, e un cuoco nuovo non deve reinventare
nulla.

Il feature store è quella dispensa. Le feature si definiscono una volta, in un
posto solo, e sia chi addestra il modello sia chi lo manda in produzione le
prende dallo stesso scaffale, con la garanzia che siano *le stesse* e
*fresche*, cioè aggiornate. In più, una feature preparata bene la riusano
dieci modelli diversi: si cucina una volta, si serve a tutti.

`````

`````{tab} Superiore

Un feature store ha tipicamente due volti. L’**offline store** conserva lo
storico completo delle feature, ottimizzato per letture massicce: è la sorgente
per costruire i dataset di addestramento. L’**online store** tiene invece
l'ultimo valore di ogni feature, ottimizzato per letture a bassissima latenza:
è ciò che il servizio interroga quando deve rispondere in millisecondi. La
stessa *definizione* alimenta entrambi, ed è questa condivisione a sradicare
alla radice le incoerenze tra addestramento e produzione.

Il problema tecnico più insidioso che un feature store deve garantire è la
**point-in-time correctness**. Costruendo un esempio di addestramento
etichettato al tempo $t$, le sue feature vanno calcolate con i soli dati
disponibili *prima* di $t$: usare un valore aggregato che include informazione
successiva a $t$ inietta nel modello una conoscenza del futuro che in
produzione non avrà mai (una forma di *data leakage* temporale che gonfia le
metriche in laboratorio e crolla sul campo). Un *point-in-time join* corretto
è tedioso da implementare a mano ed è una delle ragioni per cui il feature
store esiste come componente dedicato.

`````

## Training–serving skew

Arriviamo così al guasto più classico e più costoso di tutta la disciplina,
quello che il feature store esiste per prevenire. Succede quando una feature
viene calcolata in un modo mentre il modello impara e in un modo *anche solo
leggermente diverso* mentre risponde. Il modello, tarato sui numeri del primo
calcolo, si trova davanti numeri che vogliono dire un'altra cosa, e sbaglia in
silenzio: nessun messaggio d'errore, nessun programma che si ferma.

Il nome inglese è **training–serving skew**, cioè «lo storto fra addestramento
e servizio»: *skew* è la sbilenchezza, lo scostamento fra due cose che
dovrebbero coincidere.

`````{tab} Elementare

È come tarare una bilancia in grammi e poi, senza dirlo a nessuno, pesarci
sopra in once. Nessun errore lampeggia sullo schermo: i numeri arrivano,
sembrano plausibili, e il risultato è semplicemente sbagliato.

Il caso da manuale è la **normalizzazione**, il gesto con cui si porta ogni
variabile su una scala confrontabile prima di darla al modello. Invece del
valore grezzo, al modello si dice quanto quel valore sta sopra o sotto la
media di tutti gli altri: così un importo in euro e un'età in anni diventano
paragonabili. La media va calcolata una volta sola, sui dati di addestramento,
e poi congelata: fa parte del modello quanto i pesi.

In produzione, per una svista, qualcuno la ricalcola sul *singolo lotto*
appena arrivato, ed è lì che il pavimento cede. In addestramento «alto» voleva
dire «più della media di tutti»; se la media la si rifà sul gruppetto appena
arrivato, e quel gruppetto è fatto di soli importi alti, nessuno di loro è più
sopra la media: sono tutti normali. Il modello smette di insospettirsi proprio
del lotto più sospetto che gli sia mai capitato. Sotto ci sono le due versioni affiancate, la corretta e la bacata: la
differenza è una riga sola.

`````

`````{tab} Superiore

Sia $x$ una feature e $z = (x - \mu)/\sigma$ la sua versione standardizzata,
dove $\mu$ e $\sigma$ sono media e deviazione standard. La regola vincolante è
che $\mu$ e $\sigma$ siano **statistiche del training**, stimate una volta e
*congelate*: fanno parte del modello tanto quanto i pesi. Usarle in
addestramento e ricalcolarle in produzione su un altro campione viola
l'ipotesi sotto cui il modello è stato ottimizzato. Nel codice qui sotto la
versione corretta applica $\mu,\sigma$ del training; quella bacata ricalcola
$\mu_{\text{batch}},\sigma_{\text{batch}}$ sul lotto corrente, azzerandone di
fatto la media: un batch anomalo (tutto di importi alti) viene ricondotto a
zero e il modello non lo riconosce più come anomalo.

`````

```python
import numpy as np

rng = np.random.default_rng(0)

# --- addestramento ---
# una sola feature: l'importo di una transazione, in euro
X_train = rng.normal(loc=100.0, scale=20.0, size=10_000)

# statistiche "congelate" al momento dell'addestramento
mu, sigma = X_train.mean(), X_train.std()

# modellino lineare gia' addestrato sulla feature normalizzata z = (x - mu)/sigma
# lo score passa in una sigmoide -> probabilita' che la transazione sia una frode
w, b = 1.5, -0.2


def sigmoide(t):
    return 1.0 / (1.0 + np.exp(-t))


def predici_corretto(x):
    z = (x - mu) / sigma          # normalizza con le statistiche DEL TRAINING
    return sigmoide(w * z + b)


def predici_bacato(x_batch):
    # BUG: normalizza con media/std DEL BATCH corrente, non del training
    z = (x_batch - x_batch.mean()) / x_batch.std()
    return sigmoide(w * z + b)


# in produzione arriva un batch anomalo: importi molto piu' alti del solito
X_prod = rng.normal(loc=160.0, scale=20.0, size=32)

p_ok = predici_corretto(X_prod)
p_bug = predici_bacato(X_prod)

print("prob. media di frode (corretta):", round(float(p_ok.mean()), 3))
print("prob. media di frode (bacata):  ", round(float(p_bug.mean()), 3))
print("scarto massimo sulle predizioni:", round(float(np.abs(p_ok - p_bug).max()), 3))
```

```text
prob. media di frode (corretta): 0.967
prob. media di frode (bacata):   0.454
scarto massimo sulle predizioni: 0.793
```

Lo stesso identico modello, sugli stessi identici dati, dà due risposte
opposte. La pipeline corretta riconosce il lotto come sospetto: probabilità
media di frode $0{,}97$, cioè un allarme netto. Quella bacata, ricentrando ogni
lotto su sé stesso, cancella l'anomalia e scende a $0{,}45$, che non vuol dire
«innocuo»: vuol dire **testa o croce**, ed è anche peggio, perché un sistema
antifrode tarato per intervenire sopra una certa soglia adesso lascia passare
tutto senza fiatare. Su singole transazioni la differenza fra le due risposte
arriva a $0{,}79$. Nessun errore, nessun avviso: solo predizioni sbagliate.
Ecco perché la definizione di una feature deve vivere in *un posto solo*,
condiviso tra addestramento e servizio: è il compito del feature store.

## Validare i dati in ingresso

Il *training–serving skew* è un bug di *calcolo* delle feature. Ma un'intera
famiglia di guai arriva prima, dal dato grezzo stesso: un campo che cambia
unità di misura, una colonna che di colpo si riempie di valori nulli, un codice
prodotto che non era mai comparso. Nel software normale un input malformato fa
esplodere il programma, e l'errore si nota subito. Un modello, invece, un numero
lo restituisce *sempre*: dagli in pasto spazzatura e ti darà una predizione
dall'aria rispettabile. Per questo i dati in ingresso vanno **validati
esplicitamente**, come si controlla la merce alla porta del magazzino prima di
metterla a scaffale.

`````{tab} Elementare

Al ricevimento merci di un supermercato qualcuno controlla ogni bancale: è il
prodotto giusto? La quantità è quella dell'ordine? Ci sono confezioni rotte o
scadute? Chi non passa il controllo non entra. La validazione dei dati fa lo
stesso, e verifica quattro cose. Lo **schema**: ci sono tutte le colonne
attese, e del tipo giusto (un'età è un numero, non la parola «trenta»)? Il
**range**: i valori sono plausibili (un'età tra 0 e 120, un importo non
negativo)? I **valori mancanti**: quante caselle sono vuote, e possiamo
permettercelo? Le **distribuzioni**: i numeri di oggi somigliano a quelli di
ieri, come valore tipico e come quanto sono sparpagliati, e le categorie
arrivano nelle stesse proporzioni? Le prime tre si controllano su ogni singola
scheda (in gergo un *record*, ed è la parola che userà il codice qui sotto);
l'ultima solo guardando tante schede insieme, perché una scheda da sola non ha
una media.

`````

`````{tab} Superiore

La validazione dei dati è uno dei quattro assi della **ML Test Score**
{cite}`breck2017ml`, la rubrica di collaudo che misura la maturità di un
sistema di ML: include test sullo schema delle feature, sui loro intervalli e
sul fatto che ogni feature apporti davvero valore. Conviene distinguere due
livelli. I controlli **puntuali** (tipo, obbligatorietà, intervallo, assenza
di `NaN`) si applicano a ogni record isolato e sono economici: sono quelli che
implementiamo qui sotto. I controlli **distribuzionali** (la media di una
feature è slittata? la proporzione di una categoria è raddoppiata?) richiedono
di confrontare un lotto con una *baseline* di riferimento, ed è qui che la
validazione statica sfuma nel **monitoraggio** del *dataset shift*
{cite}`quinonero2009dataset`: lo abbiamo inquadrato in termini statistici
nella sezione «Quando i dati cambiano», e il suo lato operativo (sorvegliare
le distribuzioni nel tempo e decidere quando riaddestrare) avrà una sezione
dedicata. Qui restiamo al primo livello: fermare alla porta il record
palesemente malformato.

`````

```python
from math import isnan

# schema: per ogni campo, il tipo atteso, l'intervallo ammesso e se e' obbligatorio
SCHEMA = {
    "eta":     {"tipo": int,   "min": 0,   "max": 120,   "obbligatorio": True},
    "importo": {"tipo": float, "min": 0.0, "max": 1e6,   "obbligatorio": True},
    "citta":   {"tipo": str,                             "obbligatorio": True},
}


def valida(record, schema):
    """Controlla un record (dict) contro lo schema; ritorna la lista degli errori."""
    errori = []
    for campo, regole in schema.items():
        # 1) valore assente o nullo
        if campo not in record or record[campo] is None:
            if regole.get("obbligatorio"):
                errori.append(f"{campo}: valore mancante")
            continue
        valore = record[campo]
        # 2) NaN, il "buco" numerico di NumPy/Pandas
        if isinstance(valore, float) and isnan(valore):
            errori.append(f"{campo}: NaN")
            continue
        # 3) tipo sbagliato
        if not isinstance(valore, regole["tipo"]):
            atteso = regole["tipo"].__name__
            errori.append(f"{campo}: tipo {type(valore).__name__}, atteso {atteso}")
            continue
        # 4) fuori dall'intervallo ammesso
        if "min" in regole and valore < regole["min"]:
            errori.append(f"{campo}: {valore} sotto il minimo {regole['min']}")
        if "max" in regole and valore > regole["max"]:
            errori.append(f"{campo}: {valore} oltre il massimo {regole['max']}")
    return errori


records = [
    {"eta": 34, "importo": 250.0, "citta": "Milano"},          # valido
    {"eta": 200, "importo": 90.0, "citta": "Roma"},            # eta fuori range
    {"eta": 41, "importo": float("nan"), "citta": "Napoli"},   # importo NaN
    {"eta": 29, "importo": 60.0},                              # citta mancante
    {"eta": "trenta", "importo": 15.0, "citta": "Torino"},     # eta di tipo sbagliato
]

for i, r in enumerate(records):
    errori = valida(r, SCHEMA)
    print(f"record {i}:", "OK" if not errori else " | ".join(errori))
```

```text
record 0: OK
record 1: eta: 200 oltre il massimo 120
record 2: importo: NaN
record 3: citta: valore mancante
record 4: eta: tipo str, atteso int
```

Poche righe, ma è la porta blindata del sistema: ogni scheda che entra viene
promossa o respinta secondo regole esplicite, e le respinte finiscono in un
registro invece che, silenziosamente, dentro il modello.

E proprio su questo guardiano va detta una cosa, perché il guardiano ha una
falla. In Python il vero e il falso sono, sotto sotto, dei numeri: vero vale
uno e falso vale zero. Ne segue che un'età scritta «vero» passa indenne da tutti
e due i controlli, quello sul tipo (perché vero *è* un numero) e quello
sull'intervallo (perché uno sta fra zero e centoventi). Il guardiano dice che va
tutto bene, e non va bene niente.

Non è un caso di scuola: è quello che capita quando una colonna di sì e no viene
letta come vero-e-falso da un programma e come zero-e-uno da un altro, cioè
proprio la classe di guasti silenziosi che un CSV senza tipi dichiarati produce
a getto continuo. La cura è chiedere il tipo *esatto*
(`type(valore) is regole["tipo"]`) invece di accontentarsi di uno che gli
somiglia.

In un impianto vero, poi, questo schema si arricchisce (soglie su quante
caselle vuote si tollerano, controlli di coerenza fra un campo e l'altro,
l'aggancio ai controlli sulle distribuzioni) ma l'ossatura resta questa:
dichiarare cosa ci si aspetta dai dati, e verificarlo prima di fidarsene.
Trattare i dati da cittadini di prima classe significa, alla fine, esattamente
questo: dargli un contratto, e farlo rispettare.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Nel machine learning il codice sta quasi fermo e i **dati** sono la piena:
  per questo si conservano, si controllano e si sorvegliano con la stessa cura
  che di solito si riserva al programma.
- **Versionare i dati** vuol dire tenere la cronologia di una tabella come
  quella di un documento condiviso, senza duplicare montagne di file: si passa
  il dataset in un tritatutto che ne ricava un codicino, e si conserva quello.
  Se cambia anche un pixel, il codicino cambia del tutto.
- Una **pipeline** è una filiera per i dati: passaggi collegati, ognuno con un
  compito e con il suo controllo, sempre nello stesso ordine, senza ritocchi a
  mano che non lascino traccia. Altrimenti diventa una giungla di tubature che nessuno
  sa più dove portino.
- Il **formato** in cui i dati aspettano fra una stazione e l'altra non è un
  dettaglio: l'archivio **per colonna** (**Parquet**) legge solo le poche voci
  che servono invece di attraversare tutte le schede, e si comprime molto
  meglio perché mette vicini valori che si somigliano. Il CSV va bene per
  passare una tabella a una persona, non per il resto.
- Il bug più costoso del mestiere è calcolare una stessa informazione in un
  modo mentre si impara e in un modo appena diverso mentre si risponde: nessun
  errore compare a schermo, solo predizioni sbagliate. La cura è definirla in
  un posto solo (la **dispensa comune**, il *feature store*) e usarla di lì da
  tutte e due le parti.
- I dati in ingresso si **controllano alla porta**, come la merce al
  ricevimento di un supermercato: ci sono tutte le colonne? i valori sono
  plausibili? quante caselle sono vuote? Chi non passa il controllo finisce in
  un registro, non dentro il modello.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Nel ML il codice è spesso la parte stabile e i **dati** la parte viva:
  l'approccio **data-centric** sposta l'attenzione dal limare il modello al
  migliorare i dati, e li tratta come artefatti di prima classe (versionati,
  testati, sorvegliati {cite}`huyen2022designing`).
- **Versionare i dati**: `git` non basta (file grandi e binari); si usa
  l’**hash del contenuto** come indirizzo (*content-addressable storage*), da
  cui immutabilità, deduplicazione e **lineage** che lega ogni modello ai dati
  esatti che l'hanno prodotto. Gli strumenti che lo fanno cambiano; il
  meccanismo è quello con cui `git` indirizza i propri oggetti.
- Una **pipeline di dati** (estrazione → pulizia → feature → training) va
  resa **riproducibile e orchestrata** (un DAG di stadi idempotenti), per non
  degenerare nella *pipeline jungle*; automatizzarla per intero è il cuore della
  CD4ML {cite}`sato2019continuous`.
- Il **formato** in cui i dati stanno fra uno stadio e l'altro non è un
  dettaglio: un formato **colonnare** (**Parquet**) legge solo le colonne che
  servono, comprime meglio perché i valori simili sono vicini (misurato:
  diciotto volte su colonne categoriche, poco più di due su float casuali, fra
  due e tre su una tabella mista, e due sole su istanti ordinati finché non si
  chiede la *delta encoding*, che porta a quaranta), salta interi blocchi grazie alle
  statistiche, e ha uno **schema con i tipi** che al CSV manca. **Arrow** fa la
  stessa cosa **in memoria**, e serve a passarsi una tabella fra processi o
  linguaggi senza convertirla. CSV per un umano, Parquet per tutto il resto.
- Il **feature store** centralizza la definizione delle feature: stessa ricetta
  in addestramento (*offline*) e in produzione (*online*), riuso tra modelli,
  freschezza e **point-in-time correctness** contro il *leakage* temporale.
- Il **training–serving skew** è il bug silenzioso per eccellenza: una feature
  calcolata diversamente in training e in produzione (es. normalizzare col
  batch invece che con le statistiche congelate del training) sballa le
  predizioni senza sollevare alcun errore.
- **Validare i dati in ingresso** (schema, tipi, range, valori mancanti,
  distribuzioni) è un asse della **ML Test Score** {cite}`breck2017ml`. I
  controlli puntuali fermano il record malformato; quelli distribuzionali
  sfumano nel monitoraggio del *dataset shift* {cite}`quinonero2009dataset`.
  Attenzione al controllo di tipo: `isinstance` accetta i sottotipi, e in
  Python `bool` è un `int`.
```
`````
