# Il bootstrap: quanto ci credo a questo numero

Il nome più famoso della statistica al calcolatore è stato dato in una riga
sola, e con la più svagata delle motivazioni. Bradley Efron, nel 1979, annuncia
un metodo *più elementare* del jackknife (l'attrezzo che allora si usava), e lo
battezza *bootstrap* «*for reasons which will become obvious*», per ragioni che
diventeranno ovvie {cite}`efron1979bootstrap`. Le ventisei pagine che seguono su
quelle ragioni non ci tornano più sopra: il lettore se le deve dedurre da solo.

Il senso però si indovina, ed è una vanteria: *to pull oneself up by one's
bootstraps*, tirarsi su per i lacci degli stivali, in inglese vuol dire cavarsela
da soli in una situazione da cui non si potrebbe uscire senza aiuto. È
un'immagine di impossibilità fisica, e infatti il metodo di Efron sembra fare
qualcosa di impossibile: dire quanto è affidabile una stima **senza raccogliere
un solo dato in più**.[^munch]

[^munch]: L'immagine viene spesso attribuita al barone di Münchhausen, e
    l'attribuzione è sbagliata: nel racconto di Rudolf Erich Raspe il barone si
    tira fuori dalla palude, col cavallo, afferrandosi per il **codino** della
    parrucca. Gli stivali sono un'aggiunta della tradizione americana
    dell'Ottocento, dove la frase nasce come esempio di cosa **non** si può
    fare, e solo dopo diventa l'elogio di chi si fa da sé.

## Un numero da solo non dice quanto balla

Ogni cifra stampata fin qui ne ha nascosta un'altra,
e conviene vederlo su un caso qualunque. Mettiamo che un modello dia l’$87\%$
di accuratezza: sì, ma su *questo* test. Rifacendo la prova con altri trecento
esempi, quanto verrebbe? $86\%$? $91\%$? La differenza fra i due casi decide
se conviene mettere il modello in produzione, e il numero da solo non la dice.

`````{tab} Elementare

Per alcune quantità la risposta esiste da due secoli. Se la stima è una
**media**, la statistica ha una formula che dice di quanto ci si può aspettare
che balli: si prende quanto sono sparpagliati i dati e si divide per la radice
di quanti sono. È il motivo per cui un sondaggio su mille persone dichiara un
margine di poco più di tre punti, e il giornalista può scriverlo senza rifare il
sondaggio.

Il problema è che quella formula vale per la media e per poco altro. Restano
scoperte quasi tutte le quantità che si vogliono misurare davvero: la
**mediana** degli stipendi (che è più onesta della media, perché non si fa
trascinare da tre amministratori delegati), l’**accuratezza** di un modello, il
rapporto fra due grandezze, la differenza fra le prestazioni di due modelli
messi a confronto. Per tutte queste, la formula o non esiste, o esiste sotto
ipotesi che i dati veri non rispettano.

Un attrezzo prima del 1979 c'era, e si chiama jackknife: togli un dato dal
mucchio, rifai il conto senza di lui, rimettilo a posto e passa al successivo,
fino all'ultimo. Quanto i risultati si allontanano fra loro dice quanto la stima
balla, e sulla media va bene. Sulla mediana no, e la ragione si vede a occhio:
sessanta stipendi in fila hanno la mediana al centro, togliendone uno il centro
scivola di un posto e mai di più, così il risultato è sempre uno dei due numeri
che stavano in mezzo. Sessanta prove che ridanno due soli valori fanno sembrare
la mediana molto più ferma di quanto sia, ed è proprio la mediana quella su cui
si voleva una risposta.

Per lei, e per le altre quantità scoperte, la risposta onesta era: se vuoi
sapere quanto balla, rifai l'indagine venti volte e guarda. Cioè, quasi sempre:
non lo saprai.

`````

`````{tab} Superiore

Il problema è quello classico dell'inferenza. Si osserva un campione
$\mathbf{x} = (x_1, \dots, x_m)$ estratto da una distribuzione ignota $F$ (due
avvertenze sui simboli, che nella notazione consolidata del bootstrap cambiano
mestiere: $\mathbf{x}$ è l'intero campione e non le caratteristiche di un
esempio, e $\theta$ è la quantità da stimare, non i parametri di un modello), si
calcola una statistica $\hat{\theta} = s(\mathbf{x})$, e si vuole la
**distribuzione campionaria** di $\hat{\theta}$, cioè come varierebbe
ripetendo l'estrazione da $F$. Da lì si ricavano errore standard, intervalli di
confidenza e test.

Per $\hat\theta = \bar{x}$ il teorema del limite centrale dà la risposta
asintotica, $\operatorname{se}(\bar{x}) = \sigma/\sqrt{m}$ stimabile con
$s/\sqrt{m}$. Per statistiche non lineari o non regolari (mediana, quantili,
rapporti, coefficienti di correlazione, AUC, differenza fra due metriche) la
distribuzione campionaria dipende da $F$ in modo che non si scrive in forma
chiusa; le approssimazioni con il metodo delta richiedono derivabilità e danno
comunque solo il primo ordine. Il *jackknife* di Quenouille e Tukey, che
riestima la statistica togliendo un dato per volta, è la soluzione precedente, e
sulla mediana **fallisce**, cosa già nota quando Efron scrive; quello che lui
mostra, nel paragrafo 3 dell'articolo, è che sulla stessa mediana il bootstrap
invece funziona, e quel confronto è una delle ragioni per cui il metodo nasce.

`````

## Il campione come mondo in miniatura

L'idea di Efron sta in una riga, e conviene leggerla due volte perché è più
audace di quanto sembri.

`````{tab} Elementare

Quello che vorresti fare è chiaro: rifare l'indagine mille volte, ottenere mille
mediane, e guardare quanto quelle mille sono sparpagliate. Ecco quanto balla la
tua mediana. Non puoi: hai un campione solo, e raccoglierne altri novecento
novantanove costa quanto i primi.

Efron fa questo ragionamento. Il campione che hai in mano è la miglior fotografia
che esista del mondo da cui viene: sessanta stipendi presi a caso somigliano al
paese più di qualunque altra cosa tu abbia. E allora, invece di pescare mille
campioni nuovi dal **mondo** (impossibile), peschiamo mille campioni nuovi
**dalla fotografia**.

Come si pesca da una fotografia di sessanta numeri un campione nuovo di sessanta
numeri? Rimettendo dentro. Si estrae un numero a caso, lo si segna, lo si
**rimette nell'urna**, e si ripete sessanta volte. Il campione che ne esce ha
sessanta numeri come l'originale, ma non è l'originale: qualcuno è uscito due o
tre volte, qualcun altro non è uscito affatto. Poi se ne calcola la mediana. Poi
si ricomincia: mille volte, o diecimila, che costano solo tempo di
calcolatore.

Alla fine hai un mucchio di mediane. Non vengono da indagini vere, ma il modo in
cui si sparpagliano è una stima onesta di come si sparpaglierebbero quelle vere.
Dal mucchio esce anche l'intervallo, cioè i due estremi da scrivere accanto alla
stima: metti le mille mediane in fila dalla più piccola alla più grande, scarta
le venticinque più basse e le venticinque più alte, e i due valori rimasti ai
bordi sono l'intervallo che promette di contenere la mediana vera novantacinque
volte su cento. E questo lo puoi fare stasera, con i dati che hai già.

Il punto in cui l'analogia si rompe, e va detto perché è il punto in cui il
metodo si rompe davvero: **la fotografia non può mostrare quello che non
inquadra**. Se il campione è piccolo, o storto (solo stipendi del Nord, solo
clienti soddisfatti), il bootstrap ricampiona quella stortura con la stessa
diligenza con cui ricampiona il resto, e restituisce un intervallo stretto e
sbagliato. Il bootstrap misura la variabilità del campionamento, non i peccati
del campione.

`````

`````{tab} Superiore

Si sostituisce la distribuzione ignota $F$ con la **distribuzione empirica**
$\hat{F}_m$, che mette massa $1/m$ su ciascun dato osservato. Un campione
bootstrap $\mathbf{x}^*$ è un campione di taglia $m$ estratto da $\hat{F}_m$,
cioè estratto **con reimmissione** dai dati, e la distribuzione bootstrap è
quella di $\hat\theta^* = s(\mathbf{x}^*)$ al variare del sorteggio.

Il principio è la sostituzione

$$
\underbrace{\hat\theta - \theta(F)}_{\text{ignota}}
\qquad\longleftrightarrow\qquad
\underbrace{\hat\theta^* - \hat\theta}_{\text{simulabile}} ,
$$

giustificata dal fatto che $\hat{F}_m \to F$ (Glivenko–Cantelli) e che per
statistiche sufficientemente regolari la mappa $F \mapsto$ legge di
$\hat\theta$ è continua nel punto giusto. L'errore standard si stima con la
deviazione standard delle $B$ repliche,

$$
\widehat{\operatorname{se}} = \sqrt{\frac{1}{B-1}\sum_{b=1}^{B}
\bigl(\hat\theta^{*}_{b} - \bar{\theta^{*}}\bigr)^2},
$$

e l'intervallo **percentile** al livello $1-2\alpha$ è la coppia di quantili
empirici $[\hat\theta^{*}_{(\alpha)},\, \hat\theta^{*}_{(1-\alpha)}]$.

Due avvertenze. La prima: $B$ conta le simulazioni, non gli esempi del
campione, e l'unico costo è di calcolo; $B = 200$ basta per un errore
standard, per i quantili di un intervallo ne servono almeno $1000$ e $10\,000$
non fanno male. La seconda: l'intervallo percentile è il più semplice e non il
migliore. È corretto al primo ordine, e con statistiche distorte o asimmetriche
sotto-copre; le versioni $\mathrm{BCa}$ (*bias-corrected and accelerated*) e
$t$-bootstrap correggono, al prezzo di più conti. Su dati simulati, dove il
valore vero si conosce, la sotto-copertura si può misurare.

`````

Il ricampionamento con reimmissione è esattamente la mossa con cui il
**bagging** costruisce dataset diversi avendone
uno solo, e il conto di quanti esempi restano fuori (poco più di un terzo) è
già stato fatto lì. Quello che cambia è cosa se ne fa: il bagging usa i
campioni per addestrare modelli diversi da far votare, qui li si usa per
guardare quanto balla una stima. Stesso attrezzo, due mestieri; e fra poco quel
terzo tornerà, a rovescio, a dire dove il bootstrap non arriva.

## Il conto, su sessanta stipendi

Su sessanta stipendi fabbricati apposta (con la coda a destra che hanno i
redditi veri), ricampionati diecimila volte, quelle diecimila mediane danno
due numeri: l’**errore standard**, che è quanto la stima balla in media,
e l’**intervallo** dentro cui cade nel $95\%$ dei casi. Poi fa la stessa cosa
sulla media, dove esiste anche la formula di due secoli fa, per avere qualcosa
contro cui controllarlo.

```python
import numpy as np

rng = np.random.default_rng(0)
# sessanta stipendi, con la coda a destra che hanno i redditi veri
campione = np.round(np.exp(rng.normal(np.log(28_000), 0.45, 60)))

def bootstrap(dati, stima, giri=10_000, seme=0):
    """La stima calcolata su `giri` ricampionamenti con reimmissione."""
    r = np.random.default_rng(seme)
    idx = r.integers(0, len(dati), (giri, len(dati)))
    return stima(dati[idx], axis=1)

for nome, f in (("mediana", np.median), ("media", np.mean)):
    d = bootstrap(campione, f)
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"{nome:8s} = {f(campione):9.0f}   intervallo 95%: [{lo:.0f}, {hi:.0f}]"
          f"   errore standard {d.std():.0f}")

# per la media la formula esiste: errore standard = s / radice di n
s = campione.std(ddof=1) / np.sqrt(len(campione))
print(f"\nformula per la media: errore standard {s:.0f}, "
      f"intervallo [{campione.mean()-1.96*s:.0f}, {campione.mean()+1.96*s:.0f}]")
```

```text
mediana  =     29282   intervallo 95%: [25720, 32885]   errore standard 2061
media    =     31425   intervallo 95%: [28219, 34806]   errore standard 1675

formula per la media: errore standard 1666, intervallo [28159, 34691]
```

Conviene leggere per prime le ultime due righe, perché sono il collaudo di
tutto il resto. Sulla **media**, dove la risposta si sa da due secoli, il
bootstrap dice $1675$ e la formula dice $1666$: differiscono di nove unità su
milleseicento, cioè di mezzo punto percentuale, che è il rumore delle diecimila
simulazioni. Il metodo non sta inventando niente dove qualcuno può controllarlo,
ed è per questo che ci si può fidare anche dove nessuno può: per la **mediana**
una formula non c'è, e il bootstrap risponde lo stesso, $2061$.

Da notare, per inciso, che la mediana ($29\,282$) sta ben sotto la media
($31\,425$), che è quello che succede sempre agli stipendi e alle case: pochi
valori altissimi tirano su la media e lasciano stare la mediana.

```{figure} ../figures/bootstrap-si-accumula.svg
:name: fig-bootstrap-accumula
:alt: "A sinistra i 60 stipendi del campione, sempre gli stessi, disposti in colonnine lungo un asse orizzontale. A ogni scatto alcuni di essi si accendono in terracotta perché sono stati pescati, e diventano più grandi se pescati più volte, mentre quelli rimasti fuori restano pallidi: è il ricampionamento con reimmissione. A destra un istogramma cresce di scatto in scatto, una barra per ogni mediana calcolata, da una sola pescata fino a 10.000 pescate, e prende una forma a campana stretta e quasi simmetrica. Alla fine compare sotto l'istogramma l'intervallo al 95 per cento, da 25720 a 32885, che è la risposta cercata: quanto balla la mediana."
:width: 100%

Il gesto, in movimento. A sinistra il campione, che non cambia mai: a ogni giro
alcuni dei suoi punti vengono pescati (in terracotta, più grossi se pescati più
volte) e altri restano fuori. A destra la mediana di ciascuna pescata si
aggiunge alle precedenti, e la pila che ne viene fuori è la risposta: da un
campione solo, una distribuzione.
```

Quello che {numref}`fig-bootstrap-accumula` fa vedere e la tabella no è che il
campione **non si tocca**: la variabilità che si vede a destra non viene da dati
nuovi, viene tutta dal sorteggio di quali dei sessanta guardare. È il punto in
cui il metodo sembra un imbroglio, e a togliere il sospetto è il collaudo
dell'intervallo.

### Ma quell'intervallo è davvero al 95%?

Un intervallo di confidenza al $95\%$ promette una cosa precisa e verificabile:
ripetendo tutto l'esperimento tante volte, il valore vero deve cadere dentro
l'intervallo nel $95\%$ dei casi. Qui i dati li fabbrichiamo noi, quindi il
valore vero si conosce e la promessa si può controllare.

```python
import numpy as np

MU, SIGMA, M, PROVE = np.log(28_000), 0.45, 60, 4000
VERA = np.exp(MU)          # per una distribuzione log-normale la mediana e' exp(mu)

rng = np.random.default_rng(0)
dentro = 0
for k in range(PROVE):
    c = np.exp(rng.normal(MU, SIGMA, M))          # un'indagine nuova, da capo
    d = bootstrap(c, np.median, giri=1000, seme=k)
    lo, hi = np.percentile(d, [2.5, 97.5])
    dentro += lo <= VERA <= hi

quota = dentro / PROVE
# anche questa percentuale e' una stima, e balla: ecco entro quali estremi.
# due decimali e non uno, perche' la conclusione si gioca sul secondo
margine = 1.96 * np.sqrt(quota * (1 - quota) / PROVE)
print(f"mediana vera: {VERA:.0f}")
print(f"l'intervallo la contiene {dentro} volte su {PROVE}: {quota:.2%}")
print(f"margine di queste {PROVE} prove: da {quota-margine:.2%} "
      f"a {quota+margine:.2%}")
```

```text
mediana vera: 28000
l'intervallo la contiene 3771 volte su 4000: 94.27%
margine di queste 4000 prove: da 93.56% a 94.99%
```

Il $94{,}27\%$ contro il $95\%$ promesso è la risposta giusta a due domande
diverse. Alla prima («funziona?») risponde di sì, e senza esitazioni: un metodo
che non funzionasse darebbe $70\%$ o $99\%$, non un numero a tre quarti di
punto dal bersaglio.

Alla seconda («è esatto?») risponde di no, ma la risposta va letta con la terza
riga in mano, ed è per averla che quel numero è stampato con **due** decimali
invece di uno. Anche il $94{,}27\%$ è una stima, ottenuta da quattromila prove
e non da infinite, quindi balla pure lui, fra $93{,}56\%$ e $94{,}99\%$. Il
$95\%$ promesso resta fuori da quell'intervallo, e ci resta per cinque
millesimi di punto: l'esperimento rileva la sotto-copertura, e la rileva **di
misura**.

Un decimale in meno avrebbe capovolto la conclusione senza cambiare un dato:
$94{,}3\%$ più $0{,}7$ fa esattamente $95{,}0$, e chi legge così crede che il
bersaglio sia dentro. È il motivo per cui, quando un confronto si gioca sulla
seconda cifra, la seconda cifra si stampa.

La sotto-copertura è comunque un fatto documentato dell'intervallo
**percentile**, il più semplice dei tre che Efron e i suoi successori hanno
costruito: con campioni piccoli e distribuzioni storte sotto-copre di poco e
sistematicamente. Chi ha bisogno del numero preciso usa il $\mathrm{BCa}$; chi
ha bisogno di sapere se una differenza è solida usa questo, che costa quattro
righe.

La prudenza ci ha portati in un posto preciso: per giudicare una percentuale
misurata abbiamo dovuto chiederci di quanto ballasse, e la risposta ha deciso
il verdetto. È la domanda da cui siamo partiti, applicata a noi stessi.

## Dove si rompe, e perché è lo stesso conto del bagging

Un metodo che sembra dare qualcosa in cambio di niente va provato dove non
funziona, se no non si sa dove ci si può fidare. Il caso da manuale è il
**massimo**.

```python
import numpy as np

rng = np.random.default_rng(7)
c = np.exp(rng.normal(np.log(28_000), 0.45, 60))

for nome, f in (("massimo", np.max), ("mediana", np.median)):
    d = bootstrap(c, f, giri=10_000)
    print(f"{nome:8s}: {len(np.unique(d)):4d} valori distinti su 10000 ricampionamenti")

d = bootstrap(c, np.max, giri=10_000)
print(f"\nil massimo del campione vale {c.max():.0f}")
print(f"quota di ricampionamenti che ridanno esattamente quel valore: {(d == c.max()).mean():.3f}")
print(f"1 - (1 - 1/m)^m con m=60 vale                               : {1-(1-1/60)**60:.3f}")
```

```text
massimo :    9 valori distinti su 10000 ricampionamenti
mediana :  160 valori distinti su 10000 ricampionamenti

il massimo del campione vale 68882
quota di ricampionamenti che ridanno esattamente quel valore: 0.631
1 - (1 - 1/m)^m con m=60 vale                               : 0.635
```

Diecimila ricampionamenti e **nove** risposte diverse: la distribuzione bootstrap
del massimo si riduce a un mucchietto di nove valori, e nei due
terzi dei casi è sempre lo stesso. La ragione è ovvia una volta detta: il massimo
di un ricampionamento non può superare il massimo del campione, quindi da quel
lato l'intervallo è murato, e dall'altro può solo saltare al secondo, al terzo, al
quarto valore più grande. Non c'è niente da guardare, perché la statistica
dipende da un dato solo.

E le ultime due righe sono il pezzo che conviene portarsi via. Il $0{,}631$
contato sui diecimila ricampionamenti e il $0{,}635$ che la formula dà per
$m = 60$ sono lo stesso numero, ed è il conto di
{doc}`Alberi e metodi ensemble <alberi-ensemble>`
letto al contrario. Là si contavano gli esempi che restano **fuori** da un
campione bootstrap, poco più di un terzo, e la notizia era buona: su quel
terzo si misura l'errore gratis. Qui si contano gli altri, i due terzi che
restano **dentro**, e la stessa notizia diventa la condanna del metodo, perché
due ricampionamenti su tre contengono il massimo e quindi ridanno la stessa
identica risposta. È la stessa proprietà, letta dai due lati: un numero non è
mai buono o cattivo per conto suo, dipende da che cosa gli si chiede di
reggere.

Da qui la regola pratica: il bootstrap funziona per le statistiche che dipendono
**da tutti i dati un po’** (medie, mediane, quantili non estremi, coefficienti,
metriche di un modello) e fallisce per quelle che dipendono **da uno o due dati
molto** (massimo, minimo, il valore più raro).

Ci sono altri due modi di rompersi, e sono più insidiosi perché il conto esce
lo stesso e sembra buono.

- **Dati che non sono indipendenti.** Il ricampionamento tratta i dati come
  palline in un'urna, quindi intercambiabili. In una serie temporale non lo
  sono: la temperatura di oggi somiglia a quella di ieri, e mescolando le
  palline si distrugge proprio la struttura che rende la serie una serie. Il
  risultato è un intervallo **troppo stretto**, cioè una fiducia che non c'è.
  Il rimedio si chiama *block bootstrap*, e ricampiona pezzi di serie interi
  invece che singoli valori; il capitolo sulle serie temporali torna sul perché
  quei dati vadano trattati a parte.
- **Il campione stesso.** Come diceva l'analogia della fotografia, il bootstrap
  misura la variabilità dovuta al **caso del campionamento** e nient'altro. Un
  campione raccolto male dà un intervallo stretto attorno al numero sbagliato, e
  la strettezza è la parte pericolosa, perché somiglia alla precisione.

## A che serve, quando si valuta un modello

Il posto in cui questo attrezzo torna utile subito è la valutazione dei modelli.
Un'accuratezza dell’$87\%$ misurata su duecento esempi di test e una misurata su
ventimila sono due numeri scritti uguale che valgono in modo diverso, e il
bootstrap sul test set dice quanto: si ricampionano gli esempi di test con
reimmissione, si ricalcola la metrica ogni volta, e si guardano i percentili. Se
l'intervallo del modello A e quello del modello B si accavallano per metà, la
classifica fra i due non c'è, per quanto le due cifre siano diverse.

È anche il modo giusto di leggere le classifiche pubblicate: due sistemi separati
da mezzo punto su un test da mille esempi sono, quasi sempre, la stessa cosa.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Una stima è un numero, e un numero da solo non dice **quanto balla**. Per la
  media una formula esiste da due secoli; per la mediana, per l'accuratezza di
  un modello, per un rapporto, no.
- Il **bootstrap** ricampiona i dati che hai, **con reimmissione** e nella
  stessa quantità, mille volte, e guarda quanto la stima si sparpaglia fra i
  mille. È l'unica cosa che si può fare senza raccogliere altri dati.
- Si controlla dove la risposta si sa già: sulla media il bootstrap dà $1675$ e
  la formula $1666$. Ed è per questo che ci si fida di lui sulla mediana, dove
  la formula non c'è.
- La promessa di un intervallo al $95\%$ si può misurare, e su quattromila
  prove esce $94{,}27\%$. Anche quel numero ha il suo margine (da $93{,}56$ a
  $94{,}99$), e il $95\%$ resta appena fuori: il metodo funziona, e copre un
  filo meno di quanto promette.
- **Non funziona per il massimo**: diecimila ricampionamenti danno nove risposte
  distinte, perché il massimo dipende da un dato solo. Vale per tutte le
  statistiche che poggiano su uno o due dati.
- Il campione è una **fotografia**, e una fotografia non mostra quello che non
  inquadra: su dati raccolti male il bootstrap dà un intervallo stretto attorno
  al numero sbagliato, e la strettezza è la parte pericolosa.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Il bootstrap stima la distribuzione campionaria di $\hat\theta = s(\mathbf{x})$
  sostituendo a $F$ la distribuzione empirica $\hat{F}_m$ e simulando: campioni
  di taglia $m$ **con reimmissione**, la statistica ricalcolata su ciascuno
  {cite}`efron1979bootstrap`.
- Errore standard = deviazione standard delle $B$ repliche; intervallo
  **percentile** = quantili empirici. $B \approx 200$ per un errore standard,
  $\ge 1000$ per i quantili.
- Il percentile è corretto **al primo ordine** e sotto-copre con statistiche
  distorte o asimmetriche: $94{,}27\%$ contro il $95\%$ nominale
  su $4000$ prove, con intervallo Monte Carlo $[93{,}56;\ 94{,}99]$ che
  **esclude** il valore nominale per un centesimo di punto. Il verdetto dipende
  dalla seconda cifra decimale, che va quindi stampata. $\mathrm{BCa}$ e
  $t$-bootstrap correggono.
- **Condizioni di validità**: statistica sufficientemente regolare in $F$ e dati
  **i.i.d.** Cade per statistiche di bordo (massimo, minimo: distribuzione
  bootstrap degenere, $9$ valori distinti su $10^4$ repliche) e per dati
  dipendenti (serie temporali: intervalli troppo stretti; serve il *block
  bootstrap*).
- La probabilità che un dato compaia in un campione bootstrap è
  $1 - (1-1/m)^m \to 1 - e^{-1} \approx 0{,}632$: è lo stesso conto che nel
  **bagging** produce il terzo di esempi *out-of-bag*, e qui è la ragione per cui
  il bootstrap del massimo non funziona.
- Uso in ML: intervalli attorno a una metrica misurata su un test set finito, e
  confronto fra due modelli. Intervalli accavallati vuol dire nessuna classifica.
```

`````

Con questo il capitolo ha in mano l'attrezzo che gli mancava per leggere i propri
numeri. Fin qui abbiamo imparato a costruire modelli e a misurarli; da adesso
sappiamo anche dire quando due misure sono davvero diverse e quando sono la
stessa misura scritta due volte. È una domanda che tornerà ogni volta che il
libro metterà due modelli uno accanto all'altro. La sezione che viene adesso
cambia registro del tutto: invece di misurare un confine già tracciato, va a
cercare quello **più largo** possibile, e la risposta viene da un ragionamento
di geometria.
