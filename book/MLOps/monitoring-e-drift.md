# Sorvegliare un modello vivo

Nella sezione «Quando i dati cambiano» avevamo chiuso con un'immagine: un
modello in produzione va trattato «come un impianto, non come un quadro
appeso». Un quadro, una volta appeso, non chiede più niente a nessuno; un
impianto invece vive, consuma, si scalda, si stara, e va sorvegliato con una
sala di controllo piena di spie e manometri. Lì l'immagine era servita a dire
*perché* serve monitorare. Restava tutto il seguito: quali strumenti montare
su quell'impianto, dove piazzare le spie, a che soglia farle scattare e cosa
fare quando una si accende. È il mestiere di questa sezione: il lato operativo
di un problema che nel capitolo di Machine Learning avevamo posto in termini
statistici.

Il guaio dei modelli, rispetto a un impianto industriale, è che quando si
guastano non fanno rumore. Una pompa che si rompe fischia, perde, si ferma; un
modello che ha smesso di capire il mondo continua a rispondere con la stessa
prontezza e la stessa aria sicura di sempre: solo che le risposte, poco alla
volta, diventano sbagliate. Il monitoraggio è l'orecchio che sostituiamo al
fischio che manca.

## Che cosa si misura

La prima domanda è banale solo in apparenza: che cosa mettiamo, esattamente,
sui manometri? Le cose da tenere d'occhio stanno su **tre livelli**, che vanno
dal più facile e immediato al più prezioso e lento. Impararli separati è
importante, perché ciascuno risponde a una domanda diversa e ha tempi diversi.

`````{tab} Elementare

Pensa al cruscotto della tua auto: ha tre famiglie di indicatori, e ti dicono
cose diverse.

Il **primo quadrante** sono le spie di base: motore acceso, temperatura,
livello della benzina. Ti dicono se la macchina *funziona come macchina*: si
accende, non fuma, risponde all'acceleratore. Per un modello è la stessa cosa:
il servizio è vivo? Risponde in fretta? Ogni tanto va in errore? Queste spie
si accendono in un istante e non hanno bisogno di sapere niente di *dove* stai
andando.

Il **secondo quadrante** ti dice come stai guidando *adesso*: che tipo di
strada è, quante curve, quanto vai piano. Per un modello: che tipo di
richieste stanno arrivando, e che tipo di risposte sta dando. Se ieri gli
arrivavano email lunghe in media 80 parole e oggi ne arrivano da 200, o se
ieri segnalava spam il 20% dei messaggi e oggi il 45%, il quadrante te lo
mostra subito: anche se non sai ancora se sia un bene o un male.

Il **terzo quadrante** è il più importante e il più lento: *sei arrivato dove
volevi?* Lo scopri solo alla fine del viaggio, confrontando dove sei con dove
volevi andare. Per un modello è la qualità vera: aveva ragione? E la risposta
giusta (l'utente ha davvero cliccato, il paziente era davvero malato) spesso
arriva con giorni o settimane di ritardo. A volte non arriva mai.

`````

`````{tab} Superiore

I tre livelli corrispondono a due famiglie di metriche molto diverse per natura
e per latenza {cite}`breck2017ml`.

1. **Metriche di sistema** (salute del servizio). Sono le stesse dell'ingegneria
   dei sistemi distribuiti: **latenza** (di norma i percentili, $p_{50}$ e
   soprattutto $p_{99}$, non la media, che nasconde le code lente), **tasso di
   errore** (risposte $5xx$, eccezioni, timeout), **throughput** e **uptime**.
   Non dicono nulla sulla *correttezza* delle predizioni, ma sono disponibili in
   tempo reale e sono la prima cosa che si rompe.

2. **Proprietà statistiche di input e output**. Le distribuzioni delle
   *feature* in ingresso e delle predizioni in uscita: media, varianza,
   quantili, frazione di valori mancanti per ogni *feature*; e, per un
   classificatore, il **tasso di ciascuna classe predetta**
   $\hat{p}_c = \frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[\hat{y}_i = c]$, dove $N$ è il
   numero di richieste nella finestra e $\hat{y}_i$ la classe predetta per la
   $i$-esima. Non richiedono le etichette vere: si calcolano sul traffico così
   com'è, e sono l'**allarme anticipato** del drift.

3. **Qualità vera** (metriche di modello: accuratezza, F1, calibrazione,
   errore di regressione). Sono ciò che davvero ci interessa, ma richiedono le
   **etichette vere**, e qui sta il problema strutturale del **label delay**
   {cite}`huyen2022designing`: l'etichetta arriva in ritardo (il rimborso del
   prestito si scopre a mesi, la diagnosi confermata a settimane) o non arriva
   affatto. La qualità vera è quindi una metrica *ritardata*, e per questo i
   proxy statistici del livello 2 non sono un ripiego ma una necessità: sono
   l'unica spia che si accende *prima* che il danno sia misurabile.

`````

Questo affina i tre controlli abbozzati nella sezione «Quando i dati cambiano»
(ingressi, uscite, errore) e aggiunge sotto di essi lo strato più prosaico e
più spesso dimenticato: un modello può servire predizioni perfette e restare
inutile perché risponde in tre secondi quando l'utente ne aspetta uno, o
perché va in errore su un input malformato che nessuno aveva previsto. La
correttezza è inutile se il servizio è morto.

## Rilevare il drift in pratica

Il secondo livello (la sorveglianza delle distribuzioni) è quello dove si
gioca la partita del *drift*. Nella sezione «Quando i dati cambiano» abbiamo
classificato i modi in cui il mondo può divergere dall'addestramento
{cite}`quinonero2009dataset`, e non li ripetiamo qui; bastano i loro nomi, con
accanto che cosa vuol dire ciascuno. Il **covariate shift** è quando cambia il
*tipo di richieste che arrivano*: arriva altra gente, con altre
caratteristiche. Il **label shift** è quando cambiano le *proporzioni delle
risposte giuste*: le frodi erano una su cento e adesso sono una su dieci. Il
**concept shift** è il più insidioso: le richieste sembrano identiche, ma è
cambiata *la regola* che lega la richiesta alla risposta giusta. Qui ci
interessa il gesto operativo: come ci si *accorge* che uno di questi è in
corso, in produzione, mentre accade.

Lo strumento l'abbiamo già incontrato: il **classificatore-detective**. Si
addestra un modello a distinguere i dati di ieri da quelli di oggi, e si
guarda quanto ci riesce. Il numero con cui si misura quanto ci riesce è
l'**AUC**, incontrata nel capitolo sul machine learning parlando di metriche, e
qui va letta così: vale $1$ quando il detective indovina sempre da quale delle
due finestre viene un dato, e vale $0{,}5$ quando sta tirando a indovinare,
perché fra due possibilità chi tira a caso ne azzecca la metà. Un'AUC vicina a
$0{,}5$ dice quindi che le due finestre sono indistinguibili *per lui*: è una
rassicurazione, non una prova, perché un cambiamento su una *feature* che il
detective non guarda gli passa sotto il naso. Lì era una diagnosi *una tantum*;
per l'impianto in produzione va reso una **sorveglianza continua**, e questo
richiede tre decisioni operative.

`````{tab} Elementare

Il detective è come il metal detector all'aeroporto: non deve sapere *che
cosa* porti in valigia, gli basta accorgersi che qualcosa è diverso dal solito
e far scattare un bip. Ma un metal detector va tarato con giudizio. Se è
troppo sensibile suona per la fibbia della cintura di tutti, e dopo il decimo
falso allarme le guardie smettono di dargli retta, che è il modo peggiore di
fallire. Se è troppo sordo lascia passare il coltello. Tararlo bene significa
scegliere la soglia giusta: abbastanza alta da non suonare per ogni respiro
del mondo, abbastanza bassa da non perdere il cambiamento vero. E quando
suona, serve una seconda ispezione che dica *dove* (quale tasca, quale
*feature*) è cambiato qualcosa, altrimenti il bip da solo non aiuta a
decidere.

`````

`````{tab} Superiore

In notazione, le tre famiglie richiamate qui sopra sono il *covariate shift*
($P(X)$ che cambia, con $P(y \mid X)$ invariata), il *label shift* ($P(y)$ che
cambia, con $P(X \mid y)$ invariata) e il *concept shift* ($P(y \mid X)$ che
cambia). Le tre decisioni operative sono:

- **Finestre temporali**. Si fissa una **finestra di riferimento** (un periodo
  in cui il modello era sano, spesso i dati di addestramento o un mese
  «buono») e la si confronta con una **finestra corrente** che scorre: le
  ultime $N$ richieste, o le richieste dell'ultimo giorno. Finestre corte
  reagiscono in fretta ma sono rumorose; finestre lunghe sono stabili ma
  lente.
- **Soglia sull'indicatore**. L'AUC del detective va da circa $0{,}5$
  (finestre che quel classificatore non distingue) a $1$ (perfettamente
  separabili); per rumore campionario, senza alcuno shift, oscilla attorno a
  $0{,}5$, anche sotto. Si sceglie quindi una soglia (per esempio $0{,}65$)
  oltre la quale scatta l'allarme, calibrata sul tasso di **falsi allarmi**
  accettabile.
- **Test per singola *feature***. Il detective è un test *multivariato*: dice
  *se* qualcosa è cambiato, non *cosa*. Per localizzare si affianca un test
  *univariato* colonna per colonna, tipicamente il test di
  **Kolmogorov–Smirnov** a due campioni, la cui statistica è la massima
  distanza verticale tra le due funzioni di ripartizione empiriche,

  $$
  D = \sup_x \left| F_{\text{rif}}(x) - F_{\text{cur}}(x) \right|,
  $$

  dove $F_{\text{rif}}$ e $F_{\text{cur}}$ sono le CDF empiriche della
  *feature* nella finestra di riferimento e in quella corrente. Un'alternativa
  diffusa, basata sugli istogrammi, è il *Population Stability Index*.

  Sul criterio di allarme conviene essere espliciti, perché il riflesso
  abituale qui è quello sbagliato. Alle taglie di una finestra di produzione
  (migliaia di record) il KS ha una potenza enorme e rifiuta l'ipotesi nulla su
  scostamenti che nessun modello sente: uno spostamento di due decimi di
  deviazione standard, su $n = 2000$ per finestra, dà $p \sim 10^{-7}$ in
  praticamente ogni finestra. Il problema non è la molteplicità dei test ma la
  **taglia del campione**, e correggere per Bonferroni non lo tocca, perché i
  $p$-value non sono al limite, sono a molti ordini di grandezza sotto
  qualunque soglia. L'allarme va quindi fondato sull'**ampiezza** ($D$, o una
  distanza normalizzata, o il PSI) con una soglia decisa sul significato
  pratico, tenendo il $p$-value al più come filtro contro il rumore delle
  finestre piccole. La correzione per test multipli serve contro la
  molteplicità, non contro l'eccesso di potenza, che alle taglie di produzione
  è il problema dominante.

`````

La deriva è una delle poche cose di questo libro che **in un fotogramma non si
vede**: un istogramma di oggi, da solo, non è né normale né anomalo, e lo
diventa solo accanto a quello di prima. In {numref}`fig-deriva-ks` ci sono sei
mesi di una stessa *feature*, con la finestra di riferimento ferma e quella
corrente che le scivola via.

```{figure} ../figures/deriva-ks.svg
:name: fig-deriva-ks
:alt: "Una curva cumulativa color teal sta ferma; una curva terracotta, che all'inizio le sta sopra esattamente, scivola verso destra mese dopo mese. Un segmento verticale ocra unisce le due curve nel punto in cui sono più distanti, e la scritta sotto dice di quanto: si parte da zero e si arriva a 0,40, ben oltre la soglia di 0,10."
:width: 92%

Le due finestre in cumulata, mese per mese. Il segmento verticale non è un
ornamento: **è** la statistica $D$, cioè il punto in cui le due curve si
allontanano di più. Il numero sotto è quello che decide se suonare l'allarme.
```

Due cose che {numref}`fig-deriva-ks` mostra e una formula non dice. La prima è
*dove* cade il massimo: non agli estremi, dove le due cumulate tornano
comunque a coincidere per costruzione, ma nel mezzo, e per due normali sfalsate
esattamente a metà strada fra le due medie. La seconda è che la soglia
disegnata è sull'**ampiezza**, non sul $p$-value: qui $0{,}10$, decisa sul
significato pratico. Con duemila osservazioni per finestra, la taglia presa a
esempio poche righe più su, il test respingerebbe l'ipotesi nulla già al mese
1, dove $D$ vale sei centesimi: il valore critico al cinque per cento scende a
$0{,}043$, e su venti coppie di finestre simulate il rifiuto arriva
diciannove volte, con $p$ dell'ordine di $10^{-5}$. Alla stessa deriva, con
cinquecento osservazioni, il rifiuto arriva otto volte su venti. Non è cambiato
il mondo fra i due casi: è cambiata la taglia del campione.

Mettiamo insieme le tre decisioni in poche righe eseguibili. Il codice confronta
una finestra di riferimento con una corrente, in cui iniettiamo di proposito uno
shift su una sola *feature*: calcola l'AUC del detective come indicatore globale,
stampa un allarme se supera la soglia, e in caso di allarme usa un KS per colonna
per dire *quale* *feature* è cambiata.

```python
import numpy as np
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

rng = np.random.default_rng(0)

# Finestra di riferimento (il "passato" su cui il modello e' tarato)
# e finestra corrente (le ultime richieste arrivate in produzione).
n, d = 2000, 4
riferimento = rng.normal(0.0, 1.0, size=(n, d))
corrente = rng.normal(0.0, 1.0, size=(n, d))
corrente[:, 1] += 1.2   # drift iniettato solo sulla feature 1

def punteggio_drift(rif, cur):
    """AUC del detective: quanto e' facile distinguere le due finestre."""
    X = np.vstack([rif, cur])
    y = np.hstack([np.zeros(len(rif)), np.ones(len(cur))])
    detective = HistGradientBoostingClassifier(random_state=0)
    return cross_val_score(detective, X, y, cv=5, scoring="roc_auc").mean()

SOGLIA = 0.65  # AUC oltre la quale scatta l'allarme
auc = punteggio_drift(riferimento, corrente)
print(f"AUC detective = {auc:.3f}")

if auc > SOGLIA:
    print(f"ALLARME: drift rilevato (AUC {auc:.3f} > {SOGLIA})")
    # Localizziamo: un test di Kolmogorov-Smirnov per ogni feature.
    for j in range(d):
        stat, p = ks_2samp(riferimento[:, j], corrente[:, j])
        sospetta = "  <-- sospetta" if p < 0.01 else ""
        print(f"  feature {j}: KS={stat:.3f}  p={p:.1e}{sospetta}")
else:
    print("Nessun drift rilevabile: il detective non distingue le finestre.")
```

L'output stampa `AUC detective = 0.775`, ben oltre la soglia di $0{,}65$:
l'allarme scatta. Poi il controllo tasca per tasca punta senza esitazioni la
*feature* 1 e lascia innocenti le altre tre. Il controllo è il test di
Kolmogorov–Smirnov, `KS` nel codice, che misura di quanto le due distribuzioni
si scostano (qui $0{,}45$), accompagnato dal $p$-value, che dice quanto sarebbe
improbabile vedere uno scostamento simile per puro caso (qui: minuscolo).

È lo scheletro di un sistema di monitoraggio reale, e la stessa funzione,
girata a ogni ora sulla finestra scorrevole, produce una serie storica
dell'indicatore di drift su cui si possono appendere gli allarmi. Ha però due
semplificazioni didattiche che in un impianto vero si pagano care, e vale la
pena nominarle proprio perché il codice è breve e viene copiato.

La prima è il cancello: qui i test per singola *feature* girano **solo se**
l'indicatore globale ha superato la soglia. Comodo da leggere, pericoloso da
copiare. Un cambiamento concentrato su una colonna sola fra molte può lasciare
l'indicatore globale appena sotto soglia, e allora il programma non guarda
nessuna colonna e stampa che va tutto bene: il silenzio più costoso possibile,
perché è un silenzio *dichiarato*. Un impianto vero calcola sempre i test per
colonna e tratta l'indicatore globale come uno fra gli indizi, non come
l'interruttore che decide se guardare.

La seconda è che il controllo colonna per colonna è necessario ma non
sufficiente. Guarda una colonna alla volta, quindi è cieco ai cambiamenti che
vivono nel **rapporto** fra le colonne: se altezza e peso continuano ciascuna a
distribuirsi come prima ma smettono di crescere insieme, ogni singola colonna
risulta innocente mentre il detective, che le guarda insieme, grida. Quando
capita questo, la conclusione giusta non è «falso allarme»: è che il
cambiamento sta nelle dipendenze, e va cercato con strumenti che le guardino
(le importanze del detective stesso, le correlazioni a coppie).

Una cautela finale, la stessa della sezione statistica ma più severa di come la
si racconta di solito. Il detective è addestrato **sui soli ingressi**: quello
che rileva è che è cambiato il tipo di richieste che arrivano, e nient'altro.
Non distingue un cambiamento innocuo da uno che rovina le predizioni; e non
distingue nemmeno le tre famiglie fra loro, perché anche un puro cambio di
proporzioni fra le risposte giuste (il *label shift*) sposta il tipo di
richieste in arrivo e lo fa suonare. Del **concept shift** puro, poi, non vede
niente: lì gli ingressi restano identici ed è la regola giusta a essere
cambiata sotto. Per separare i tre casi non c'è scorciatoia: servono le
etichette vere del terzo livello, o almeno le predizioni aggregate. Il
monitoraggio statistico è un allarme precoce, non un verdetto.

## Rispondere al drift

Una spia che si accende non è ancora una decisione. La risposta al drift si
organizza come una **piramide**, dal gesto più economico e automatico al più
costoso e delicato, e la regola d'oro è che la risposta sia *proporzionata* alla
prova: la maggior parte degli allarmi non deve arrivare in cima.

Alla base c'è l'**allarme**: automatico, a costo quasi nullo, tanto abbondante
quanto lo consente una buona soglia. Sopra c'è l'**indagine**: un umano guarda
*quale* *feature* è cambiata e prova a capire se è un artefatto (un sensore
rotto, un bug nella *pipeline* dei dati; più spesso è questo che un vero
mutamento del mondo), un covariate shift benigno o l'inizio di un concept
shift. Solo se l'indagine conferma un degrado reale si sale al **retraining**:
riaddestrare su dati recenti. E in cima, riservato all'emergenza, il
**rollback**: tornare in fretta al modello precedente, che è possibile solo se
i modelli sono versionati e il vecchio è ancora a portata di mano.

`````{tab} Elementare

È la spia dell'olio che si accende sul cruscotto. Non stacchi il motore al
primo lampeggio: prima *guardi* (l'allarme), poi ti fermi a *controllare* il
livello (l'indagine), poi semmai *rabbocchi o cambi l'olio* (il retraining), e
solo se il motore comincia a battere in testa (e cambiare l'olio non è
bastato) lo *spegni e chiami il carro attrezzi* (il rollback al modello
vecchio). Rispondere sempre col gesto più drastico è come cambiare il motore
ogni volta che si accende una spia: costoso, e spesso inutile.

C'è poi una scelta di fondo su *quando* rimettere mano al modello. Un'officina
può fare due cose: il tagliando **a scadenza fissa** (ogni diecimila chilometri,
che tu abbia problemi o no) oppure l'intervento **solo quando qualcosa si guasta**.
La prima è prevedibile e semplice; la seconda risparmia lavoro ma richiede spie
affidabili. I sistemi reali quasi sempre fanno entrambe.

`````

`````{tab} Superiore

Sul *quando* riaddestrare, due strategie {cite}`shankar2022operationalizing`:

- **Retraining periodico**: riaddestrare a cadenza fissa (giornaliera,
  settimanale) su una finestra scorrevole di dati recenti. Semplice, prevedibile,
  facile da automatizzare; è ciò che molti team fanno di default. Il costo è
  lavoro sprecato quando nulla è cambiato, e un ritardo pari all'intervallo
  quando qualcosa cambia in fretta.
- **Retraining innescato** (*triggered*): riaddestrare *quando* il monitoraggio
  supera una soglia. Reagisce a ciò che serve, ma dipende interamente dalla
  qualità degli allarmi.

Il punto delicato è l'**automazione** del retraining, che è il sogno di ogni
*pipeline* MLOps ma nasconde una trappola già incontrata: il **feedback
loop**. Se le predizioni del modello concorrono a generare i dati futuri, un
sistema di credito che nega prestiti non vedrà mai come sarebbero andati quei
clienti; un sistema di raccomandazione raccoglie clic solo su ciò che ha
deciso di mostrare, allora riaddestrare *automaticamente* su quei dati non
corregge il modello: ne **amplifica** i bias, cementandoli a ogni ciclo
{cite}`huyen2022designing`. È la stessa dinamica che aveva ingannato Google
Flu Trends, dove era anche il motore di ricerca, aggiornandosi, a cambiare i
dati che il suo stesso modello leggeva {cite}`lazer2014parable`. Un
*retraining loop* senza sorveglianza umana è un amplificatore puntato sul
proprio ingresso: prima o poi fischia. Per questo anche le *pipeline* più
automatizzate tengono un umano *nell'anello* alle soglie alte della piramide,
e valutano ogni candidato al retraining su dati **freschi e possibilmente non
contaminati** dalle scelte del modello in carica. Il caso limite, in cui il
feedback loop non è un incidente ma la struttura stessa del problema, lo
vedremo nel capitolo sui sistemi di raccomandazione.

`````

## Rilasciare senza rompere

Supponiamo che l'indagine abbia dato ragione all'allarme e il retraining abbia
prodotto un modello nuovo, che sui dati di test sembra migliore. Resta il
passo più rischioso di tutti: **sostituire** il modello vivo con quello nuovo.
Un modello che va benissimo in laboratorio può comportarsi in modo pessimo in
produzione: su input mai visti, con latenze diverse, con effetti che nessun
test offline cattura. Rimpiazzarlo di colpo per tutti gli utenti è una
scommessa che non conviene fare mai. Esistono tre modi per introdurlo in
sicurezza, ed è la stessa cassetta degli attrezzi dei rilasci graduali.

`````{tab} Elementare

Immagina di aver inventato un piatto nuovo per il tuo ristorante. Non lo metti nel
menù di colpo per tutti, rischiando di rovinare la serata a duecento clienti se
qualcosa non va. Fai una cosa più furba, in tre possibili modi.

Il primo: lo **cucini in parallelo** senza servirlo (il cuoco prepara il
piatto nuovo insieme a quello vecchio, tu lo assaggi in cucina e confronti, ma
al tavolo arriva ancora il vecchio). Nessun cliente corre rischi. Si chiama
*shadow*, cioè «in ombra».

Il secondo: lo **fai assaggiare a pochi tavoli**. Lo metti nel piatto di due
tavoli su cento, tieni d'occhio le loro facce, e se funziona allarghi a dieci, a
cinquanta, a tutti; se storcono il naso, lo ritiri e nessun danno è fatto. È il
*canary*, dal canarino dei minatori.

Il terzo: **due metà della sala**, stesso momento, piatto vecchio a una metà e
nuovo all'altra, e a fine serata conti chi ha lasciato il piatto pulito. Così sai
davvero se il nuovo è meglio, e non te lo sei immaginato. È il test *A/B*.

`````

`````{tab} Superiore

Le tre tecniche, in ordine crescente di esposizione {cite}`huyen2022designing`:

- **Shadow deployment**: il modello nuovo riceve *tutto* il traffico reale e
  produce le sue predizioni **in parallelo**, ma le sue risposte non vengono
  mai servite all'utente; si registrano e si confrontano offline con quelle
  del modello in carica. Rischio per l'utente nullo; costo: si paga il calcolo
  doppio e non si misurano gli effetti sul comportamento reale (nessuno
  *agisce* sulle predizioni ombra).
- **Canary release**: il modello nuovo serve davvero, ma solo una **piccola
  quota** del traffico (l'1%, il 5%), come il canarino che i minatori portavano
  in miniera per accorgersi del gas prima degli uomini. Si sorvegliano le
  metriche sulla quota canary e, se reggono, si aumenta gradualmente fino al
  100%; al primo segnale cattivo si torna indietro (*rollback*) avendo esposto
  pochi utenti.
- **A/B test**: due gruppi di utenti, assegnati a caso, ricevono
  contemporaneamente il modello A (vecchio) e il B (nuovo); si confronta una
  metrica di business su un orizzonte definito e si decide con un **test
  statistico**, verificando che la differenza sia significativa e non rumore
  campionario. È lo strumento per rispondere a *«il nuovo è davvero meglio?»*,
  mentre shadow e canary rispondono a *«il nuovo è sicuro da servire?»*.

Le tre non sono alternative ma un percorso: shadow per verificare che non si
rompa nulla, canary per limitare l'esposizione, A/B per decidere con rigore se
promuoverlo. Dietro tutte c'è il prerequisito che il modello sia **versionato**,
così che il *rollback* al precedente sia sempre un'operazione di un istante.

`````

Con questo l'anello si chiude e ricomincia. Il monitoraggio non è l'ultima
tappa di una linea retta: è l'occhio che, accorgendosi del drift, fa ripartire
il ciclo (indagine, retraining, rilascio graduale) e riporta all'inizio. Un
modello in produzione, l'abbiamo detto, non è un risultato da archiviare ma un
processo da tenere in vita {cite}`shankar2022operationalizing`; questa sezione
è il turno di guardia che quel processo richiede, ogni giorno, finché il
modello serve.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Un modello che si guasta **non fa rumore**: continua a rispondere con la
  stessa prontezza e la stessa aria sicura, solo che le risposte, poco alla
  volta, diventano sbagliate. Il monitoraggio è l'orecchio che sostituiamo al
  fischio che manca.
- Si guarda un cruscotto a **tre quadranti**: il servizio è vivo e risponde in
  fretta? che tipo di richieste stanno arrivando, e che risposte sta dando? e
  infine, la più importante e la più lenta, aveva ragione? L'ultima si scopre
  solo quando arriva la risposta giusta, che spesso arriva con settimane di
  ritardo e a volte non arriva mai.
- Per accorgersi che il mondo è cambiato si usa un **metal detector**: un
  programma che prova a distinguere i dati di ieri da quelli di oggi. Se ci
  riesce, qualcosa è cambiato; se tira a indovinare, no. Va tarato: troppo
  sensibile e suona per tutti, e dopo il decimo falso allarme nessuno gli dà
  più retta.
- Il metal detector dice *che* qualcosa è cambiato, non *cosa* e nemmeno *se è
  grave*. Soprattutto, non vede il caso peggiore: quello in cui le richieste
  sembrano identiche di ieri ma è cambiata la risposta giusta.
- Quando suona, si risponde **per gradi**, come con la spia dell'olio: prima si
  guarda, poi si controlla, poi semmai si riaddestra, e solo in emergenza si
  torna al modello vecchio. Rispondere sempre col gesto più drastico è come
  cambiare il motore ogni volta che si accende una spia.
- Riaddestrare da soli su dati che il modello stesso ha contribuito a produrre
  non lo corregge: **ne amplifica gli errori**, a ogni giro. Serve una persona
  nell'anello e dati freschi.
- Un modello nuovo non si accende di colpo per tutti: prima in ombra, poi a
  pochi tavoli, poi metà sala contro metà sala.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Si misura su **tre livelli**, dal più rapido al più prezioso: (1) salute del
  servizio (latenza, errori, uptime), (2) proprietà statistiche di input e
  output (distribuzioni, tasso di ciascuna classe predetta), (3) qualità vera,
  che richiede le etichette e soffre di **label delay** (arriva tardi o mai
  {cite}`breck2017ml`).
- I proxy statistici del livello 2 sono un **allarme anticipato**: si accendono
  *prima* che il degrado sia misurabile con le etichette vere.
- Il **detective** della sezione «Quando i dati cambiano» diventa sorveglianza
  continua con tre scelte operative: **finestre** (riferimento vs corrente
  scorrevole), **soglia** sull'AUC (tarata sui falsi allarmi) e **test per
  *feature*** (Kolmogorov–Smirnov) per localizzare il drift. Essendo addestrato
  sui soli ingressi, rileva un cambiamento della **marginale $P(X)$**, che
  covariate shift e label shift condividono, ed è cieco al *concept shift*
  puro.
- L'allarme si fonda sull'**ampiezza** dello scostamento, non sul $p$-value: a
  taglie di produzione il KS rifiuta su differenze che nessun modello sente. E
  il test per colonna è necessario ma non sufficiente: uno shift che vive nella
  struttura congiunta lascia tutte le marginali intatte.
- La risposta è una **piramide** proporzionata: allarme → indagine → retraining →
  rollback. Retraining **periodico** (a cadenza fissa) o **innescato** (a
  soglia); i sistemi reali fanno entrambi.
- Il **retraining automatico** su dati generati dal modello stesso amplifica i
  bias del **feedback loop** invece di correggerli: serve un umano nell'anello e
  dati freschi non contaminati {cite}`huyen2022designing`.
- Un modello nuovo si introduce **senza rompere**: **shadow** (risponde in
  parallelo, non serve), **canary** (piccola quota di traffico) e **A/B test**
  (confronto statistico), tutti poggiati sul modello **versionato** per un
  rollback immediato.
```
`````
