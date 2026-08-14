# Il costo del coordinamento

Nel 1975 Fred Brooks pubblica *The Mythical Man-Month*, il resoconto di cosa
fosse andato storto nella costruzione dell'OS/360 alla IBM. La tesi che lo ha
reso celebre è controintuitiva e brutale: aggiungere programmatori a un
progetto software già in ritardo lo fa ritardare di più. La ragione che porta
non è di psicologia aziendale, è aritmetica. Due persone hanno un canale di
comunicazione da mantenere, tre ne hanno tre, quattro sei, dieci quarantacinque:
i canali crescono come $n(n-1)/2$, cioè, quando le persone sono tante, come il
quadrato, mentre il lavoro utile cresce al più in proporzione alle persone.
Passata una certa soglia il secondo termine si mangia il primo.

Chi mette al lavoro più agenti si trova davanti un conto della stessa forma,
il quadrato, ma di origine diversa. Qui a moltiplicarsi non sono i canali fra
le persone: è la conversazione che si allunga a ogni intervento, e che ogni
agente rilegge da capo prima di parlare. Quello che si rilegge sono **token**,
cioè i pezzetti in cui un modello di linguaggio spezza il testo per digerirlo:
in italiano un token vale grosso modo mezza parola, e una pagina come questa ne
contiene sette o ottocento. I token hanno un prezzo di listino, e si pagano sia
quelli scritti sia quelli letti. Il capitolo sugli Agenti ha già detto, a
parole, che una squadra costa più di un solista e moltiplica i modi di
sbagliare. Questa
sezione trasforma quell'avvertimento in un conto: tre formule che si fanno su
un tovagliolo prima di scrivere una riga di codice. È il metodo di Enrico
Fermi, che al test Trinity, il 16 luglio 1945, lasciò cadere dei pezzetti di
carta al passaggio dell'onda d'urto e dal loro spostamento stimò a mente una
decina di chilotoni: contro i ventuno che le analisi radiochimiche avrebbero
stabilito poi (e i quasi venticinque a cui li ha portati una rideterminazione
del 2021 sui campioni di trinitite), l'ordine di grandezza giusto, e subito.
Prima di
assemblare una squadra conviene fare lo stesso: quanto costerà, quanto potrà
accelerare, quanto spesso sbaglierà.

## Il conto dei token

Il modello di riferimento è quello dei framework conversazionali, dove gli
agenti si scrivono in una **trascrizione condivisa** e ognuno, quando tocca a
lui, la rilegge dall'inizio prima di rispondere {cite}`wu2024autogen`. È la
scelta più naturale (nessuno resta indietro, tutti vedono tutto) ed è anche
quella che fa esplodere il conto, per un motivo che non c'entra nulla con
l'intelligenza dei partecipanti.

`````{tab} Elementare

Pensa a una chat di gruppo in cui, per regola, prima di scrivere un messaggio
devi rileggere *tutta* la conversazione dall'inizio. Il primo che parla legge
un messaggio (l'enunciato del compito). Il secondo ne legge due. Il decimo ne
legge dieci. Il trentaduesimo ne legge trentadue. Nessuno ha scritto messaggi
più lunghi degli altri, eppure la fatica totale di lettura non cresce come il
numero di partecipanti: cresce molto più in fretta, perché ogni messaggio
nuovo dovrà essere riletto da tutti quelli che parleranno dopo.

Il conto è quello che si fa a scuola sommando i primi numeri: $1+2+3+\dots+32$
fa $528$. Se invece i turni fossero otto, la somma $1+2+\dots+8$ farebbe $36$.
Quattro volte i turni, quasi **quindici** volte le righe lette. E siccome
quindici è quasi quattro per quattro, ecco tutto il problema: quando raddoppi
le persone attorno al tavolo non raddoppi il
costo della riunione, quasi lo quadruplichi, perché ogni nuovo partecipante
non solo parla, ma va anche ascoltato da tutti gli altri.

Fra poco, in tabella, lo stesso passaggio comparirà con un numero più mite,
quasi dieci invece di quasi quindici, e non è una svista. Quel quindici conta
solo le righe di conversazione; la bolletta vera comprende anche il pezzo fisso
che ciascuno si rilegge a ogni turno, cioè le istruzioni di partenza, che sono
sempre lunghe uguali e non si allungano mai. Una parte che non cresce, sommata a
una che cresce in fretta, smorza il rapporto fra i due totali. Quindici è la
crescita del pezzo che si gonfia, dieci è quella del conto intero.

`````

`````{tab} Superiore

Siano $N$ gli agenti, $R$ i giri di parola a testa e $T = N R$ il numero
totale di turni. Al turno $t$ la finestra di chi parla contiene il contesto
iniziale (istruzioni di sistema e definizione dei ruoli) di lunghezza $c_0$,
più la trascrizione: il messaggio che enuncia il compito e i $t-1$ interventi
già scritti, cioè $t$ messaggi di lunghezza media $\bar{m}$. I token letti
nell'intera conversazione sono quindi

$$
\text{token}(T) \;=\; \sum_{t=1}^{T}\left(c_0 + t\,\bar{m}\right)
\;=\; T\,c_0 \;+\; \bar{m}\,\frac{T(T+1)}{2},
$$

dove $c_0$ è il contesto iniziale in token, $\bar{m}$ la lunghezza media di un
messaggio e $T$ il numero di turni. Il primo addendo è lineare in $T$, il
secondo è quadratico e domina non appena $T\,\bar{m} \gg c_0$: asintoticamente

$$
\text{token}(T) \;=\; O\!\left(\bar{m}\,T^{2}\right)
\;=\; O\!\left(\bar{m}\,N^{2}R^{2}\right),
$$

perché $T = NR$. Due conseguenze da tenere ferme. La prima: l'esponente sta
sui **turni**, non sugli agenti; un sistema con due agenti molto loquaci può
costare più di uno con otto agenti che dicono una frase a testa. La seconda:
poiché $T$ è proporzionale a $N$, il costo *è* comunque quadratico nel numero
di agenti. Cresce come i canali $n(n-1)/2$ di Brooks, ma per un'altra ragione:
qui il quadrato non viene dalle coppie che si parlano, viene dalla **rilettura
cumulativa** di una trascrizione che si allunga, ed è quadratico nei turni
anche per un solista che lavori da solo abbastanza a lungo. Stessa forma,
meccanismo diverso.

Vale la pena guardare l'altro regime possibile, il *broadcast*, dove a ogni giro
tutti leggono e tutti scrivono insieme invece che a turno. Ciascuno legge
l'enunciato del compito e la trascrizione dei giri già chiusi, che cresce di
$N\bar{m}$ a giro, per un totale di

$$
\text{token}_{\text{broadcast}}(N, R) \;=\; N R\,(c_0 + \bar{m}) \;+\;
N^{2}\,\bar{m}\,\frac{R(R-1)}{2},
$$

cioè lo **stesso ordine** $O(\bar{m}N^2R^2)$ **con la stessa costante di testa**:
lo sconto sta in un termine di grado inferiore, e quindi si vede sui numeri
piccoli e sparisce al crescere dei turni. Con i numeri di questa sezione
($N = 4$, $R = 8$) fa $304.000$ token contro i
$328.000$ del turno a turno, il 7% in meno; a $R = 64$ il rapporto è già
$0{,}99$. Resta sotto uno per
ogni squadra di almeno due agenti, ma sempre meno. Attenzione a non leggerlo
come un risparmio: la ragione dello
sconto è che dentro un giro **nessuno legge gli altri**. Chi parla per $t$-esimo
nel regime a turni ha già davanti i $t-1$ interventi del giro in corso; in
*broadcast* tutti leggono la stessa trascrizione, ferma al giro precedente, e
quello che manca dal conto è esattamente l'informazione che manca a chi lavora.
Ciò che il *broadcast* compra davvero è la **latenza**, $R$ giri invece di $NR$
turni, perché gli interventi di un giro si producono in parallelo; e la compra
al prezzo di un giro di ritardo nel reagire a quello che ha appena detto un
altro.

`````

Mettiamo dei numeri: duemila token di istruzioni iniziali, che ciascuno si
rilegge sempre uguali a ogni turno, messaggi da cinquecento token, otto giri di
parola a testa. Poche righe di Python, che non fanno altro che eseguire la
somma, rendono la crescita visibile:

```python
# Costo in token di una conversazione a trascrizione condivisa:
# a ogni turno chi parla rilegge tutto cio' che e' stato detto finora.

def token_letti(turni, contesto_iniziale, messaggio_medio):
    """Token in ingresso sommati su tutti i turni: al turno t la finestra
    contiene il contesto iniziale piu' i t messaggi in trascrizione
    (l'enunciato del compito e i t-1 interventi precedenti)."""
    return sum(contesto_iniziale + t * messaggio_medio
               for t in range(1, turni + 1))


c0 = 2000   # istruzioni di sistema e definizione dei ruoli
m = 500     # lunghezza media di un messaggio
giri = 8    # giri di parola a testa

base = token_letti(giri, c0, m)  # un agente solo: il termine di paragone

print("agenti  turni  token letti  finestra finale  costo")
for agenti in (1, 2, 4, 8):
    turni = agenti * giri
    letti = token_letti(turni, c0, m)
    finestra = c0 + turni * m    # quanto e' larga la finestra all'ultimo turno
    print(f"{agenti:6d} {turni:6d} {letti:12,d} {finestra:16,d} {letti / base:6.1f}x")
```

```text
agenti  turni  token letti  finestra finale  costo
     1      8       34,000            6,000    1.0x
     2     16      100,000           10,000    2.9x
     4     32      328,000           18,000    9.6x
     8     64    1,168,000           34,000   34.4x
```

Il caso da tenere a mente è la terza riga. Quattro agenti che si parlano per
otto giri leggono **328.000 token** contro i 34.000 di un solista che lavora
otto turni: non quattro volte tanto, quasi **dieci** volte tanto. I token
*scritti*, invece, sono esattamente quattro volte (32 turni per 500 token fanno
16.000, contro gli 8 per 500 del solista, cioè 4.000): tutto
lo scarto sta sul lato della lettura, la parte del conto che nessuno guarda
perché non produce niente di visibile.

C'è un secondo effetto, meno contabile e più insidioso. All'ultimo turno la
**finestra** di chi parla (il testo che si ritrova davanti prima di rispondere,
e che è tutto quello di cui dispone) contiene 18.000 token di verbale, e
l'informazione decisiva (una
decisione presa al decimo turno, un vincolo enunciato al terzo) sta sepolta in
mezzo, ed è la posizione da cui un modello recupera peggio: quello che sta
all'inizio e alla fine di una finestra lunga pesa molto più di quello che sta in
mezzo. È il difetto che il capitolo sugli **Agenti**, nella sezione sul context
engineering, chiama *lost in the middle*. La trascrizione condivisa non è solo
cara, è anche il posto peggiore dove mettere qualcosa che deve essere ricordato.

## Il tetto di Amdahl

Il costo è metà del conto. L'altra metà è il guadagno, e anche lì c'è una
legge che non si aggira. Nel 1967 Gene Amdahl, l'architetto del System/360 di
IBM, portò a un convegno tre pagine scritte per raffreddare gli entusiasmi
verso i calcolatori a più processori: una parte del lavoro, sosteneva, resta
sequenziale comunque, e quella parte da sola basta a mettere un tetto.
L'argomento era in prosa; la formula che oggi porta il suo nome è stata
ricavata da quelle pagine dai commentatori venuti dopo, e vale per qualunque
cosa si provi a suddividere fra più esecutori. La parte di lavoro che *non si
può* spezzare mette un tetto all'accelerazione, e il tetto non dipende da
quanti esecutori si aggiungono. In una squadra di agenti quella parte è facile
da riconoscere: qualcuno deve decidere il piano prima che gli altri comincino,
qualcuno deve
ricucire i pezzi dopo che hanno finito. Piano e sintesi sono la strozzatura
all'ingresso e all'uscita.

`````{tab} Elementare

Immagina di dover consegnare un rapporto in dieci ore di lavoro. Tre di quelle
ore non si dividono con nessuno: un'ora per decidere la scaletta (finché non
c'è, nessuno può scrivere il proprio capitolo) e due ore alla fine per
rileggere tutto insieme, uniformare il tono e togliere le ripetizioni. Le
altre sette sono i capitoli, e quelle sì che si spartiscono.

Con quattro persone il lavoro dura $3 + 7/4 = 4{,}75$ ore: sei andato circa
due volte più veloce, non quattro. Con otto dura $3 + 7/8 = 3{,}875$ ore, cioè
$2{,}6$ volte più veloce: i quattro assunti in più hanno comprato meno della
metà di quello che avevano comprato i primi quattro. E con mille? Le sette ore
di scrittura svaniscono quasi del tutto, ma le tre di scaletta e rilettura
restano lì: quel rapporto non si chiuderà mai in meno di tre ore, cioè oltre
$10/3 = 3{,}3$ volte più veloce non si va. Nessun numero di collaboratori farà
mai quel rapporto in due ore.

`````

`````{tab} Superiore

Sia $s \in [0,1]$ la frazione **intrinsecamente seriale** del lavoro (quella
che va svolta da un solo esecutore, indipendentemente da quanti ce ne siano) e
$1-s$ la frazione parallelizzabile. L'ipotesi nascosta, e conviene dirla perché
è quella che si viola più spesso, è che il problema abbia **taglia fissa**: si
divide sempre lo stesso lavoro fra più esecutori, non se ne fa di più.
Normalizzando a 1 il tempo del singolo
esecutore, con $N$ esecutori il tempo è $s + (1-s)/N$ e l'accelerazione vale

$$
S(N) \;=\; \frac{1}{\,s + \dfrac{1-s}{N}\,},
\qquad
\lim_{N \to \infty} S(N) \;=\; \frac{1}{s},
$$

dove $S(N)$ è il rapporto fra il tempo con un esecutore e il tempo con $N$.
Il limite è la **legge di Amdahl**: il tetto è $1/s$ e non dipende da $N$. Con
$s = 0{,}3$ nessun numero di agenti porta oltre $3{,}33\times$, e la salita
verso il tetto è di quelle che si spengono in fretta:

| $N$ | 1 | 2 | 4 | 8 | 16 | $\infty$ |
|---|---|---|---|---|---|---|
| $S(N)$ | $1{,}00$ | $1{,}54$ | $2{,}11$ | $2{,}58$ | $2{,}91$ | $3{,}33$ |

Per arrivare al $90\%$ del tetto servono $N = 21$ agenti; per arrivare al
tetto, infiniti. Ma questa curva è un **limite superiore ottimistico**, perché
assume che coordinare non costi nulla, e il conto dei token qui sopra ha
appena mostrato che costa. Se si aggiunge una penale di coordinamento lineare,
cioè un tempo $\kappa(N-1)$ che ogni agente in più impone a tutti gli altri, il
tempo diventa $s + (1-s)/N + \kappa(N-1)$ e la curva non si limita ad
appiattirsi: **torna giù**. Il massimo si trova annullando la derivata,

$$
N^{*} \;=\; \sqrt{\frac{1-s}{\kappa}},
$$

dove $\kappa$ è il costo di sincronizzazione per agente aggiunto, espresso
nella stessa unità del tempo totale. Con $s = 0{,}3$ e $\kappa = 0{,}02$ (ogni
agente in più aggiunge il $2\%$ del tempo originario in coordinamento) si
ottiene $N^{*} = \sqrt{35} \approx 5{,}9$: l'ottimo è a **sei** agenti,
l'accelerazione massima è $1{,}94\times$ (contro un tetto di Amdahl di
$3{,}33$), e a sedici agenti si è già scesi a $1{,}55\times$, peggio che con
quattro. Il ginocchio della curva, non il tetto, è il numero che conta. E la
penale lineare è l'ipotesi generosa: se il coordinamento cresce come il
quadrato degli agenti, come suggerisce il conto dei token qui sopra, il
ginocchio si sposta ancora più a sinistra.

`````

La conseguenza pratica è una sola: si aggiungono agenti finché il pezzo
divisibile lo giustifica, e poi si smette. La domanda da farsi non è «un altro
agente aiuterebbe?» (la risposta è quasi sempre sì, di pochissimo) ma «quanto
vale l'*ultimo* agente aggiunto, rispetto a quello che costa?».

Sul rapporto in dieci ore la risposta si legge senza formule. Passare da quattro
persone a otto accorcia il lavoro da $4{,}75$ ore a $3{,}875$: si guadagna meno
di un'ora, cioè mezza volta scarsa di accelerazione in più. E se quei quattro e
quegli otto sono gli agenti della tabella di poco fa (due esempi diversi, stesse
proporzioni), la si paga con un conto che sale da dieci a
trentaquattro volte quello di un solista. Nessuna aritmetica sensata approva
quella spesa. E quella mezza volta è ancora la stima **generosa**, perché viene
dalla curva che finge che coordinarsi non costi niente: tenendo conto anche del
tempo che ogni nuovo arrivato fa perdere agli altri, gli otto non guadagnano
quasi più nulla rispetto ai quattro.

Il regime opposto (centinaia di agenti elementari con coordinamento quasi
gratuito) è quello degli sciami, e ha una sezione sua.

## Gli errori si compongono

Il terzo conto è quello che manda a picco le catene lunghe di agenti, ed è il
più semplice dei tre. Se un lavoro passa per $n$ passi in serie, e ogni passo
è corretto in modo indipendente dagli altri, la correttezza dell'insieme è il
**prodotto** delle correttezze, non la loro media.

`````{tab} Elementare

È il gioco del telefono senza fili. Ogni bambino ripete alla persona accanto
quello che ha sentito, e supponiamo che sia bravissimo: nel $95\%$ dei casi
ripete la frase esatta. Uno su venti, ogni tanto, cambia una parola. Sembra
una squadra affidabilissima.

Fai il conto e la sorpresa arriva subito. Con dieci bambini la frase arriva
intatta in fondo sei volte su dieci ($0{,}95$ moltiplicato per sé stesso dieci
volte fa $0{,}60$). Con venti, poco più di tre volte su dieci ($0{,}36$).
Nessuno è diventato meno bravo: è la fila che si è allungata. E perché una
fila di venti funzioni nove volte su dieci servirebbero bambini corretti nel
$99{,}5\%$ dei casi, un livello che nessun sistema reale raggiunge su compiti
aperti.

Ora però mettiamo a metà della fila di venti un adulto che ha in tasca il foglio
con la frase originale e la confronta prima di passarla avanti. La fila si
spezza in due tronconi da dieci, e l'errore accumulato nel primo non entra nel
secondo: non è un dettaglio organizzativo, è la differenza fra un sistema che funziona
e uno che no.

`````

`````{tab} Superiore

Sia $p$ la probabilità che un singolo passo sia **corretto**, e siano gli $n$
passi indipendenti. La probabilità che l'intera catena lo sia vale

$$
P(\text{catena corretta}) \;=\; p^{\,n},
$$

che decade in modo geometrico. Attenzione alla convenzione: nel capitolo sugli
Agenti e nella sezione sul loop engineering la stessa lettera indicava la
probabilità di *sbagliare* un passo, e la formula compariva come $(1-p)^n$; è
la stessa legge scritta dall'altro verso. Con $p = 0{,}95$:

$$
p^{10} = 0{,}599, \qquad p^{20} = 0{,}358.
$$

Dieci passi portano il $95\%$ di partenza al $60\%$; venti a poco più di un
terzo, cioè a un sistema che fallisce due volte su tre. Invertendo la formula,
per garantire $p^{20} \ge 0{,}90$ servirebbe
$p \ge 0{,}90^{1/20} \approx 0{,}9947$, cioè poco più di cinque errori ogni
mille passi: un requisito che nessun modello linguistico soddisfa su compiti
non banali. Questa, e non l'incapacità dei singoli agenti, è la ragione
**strutturale** per cui le pipeline lunghe falliscono: l'affidabilità non si
somma, si moltiplica.

La stessa formula dice però come uscirne, e la via non è «modelli migliori».
Se un controllo intercetta una frazione $r$ degli errori (il *recall* del
verificatore), la probabilità che al singolo passo nessun errore passi
inosservato sale a

$$
p' \;=\; 1 - (1-p)(1-r),
$$

dove $1-p$ è la probabilità che il passo sbagli e $1-r$ quella che il
controllo non se ne accorga. Attenzione a che cosa misura $p'$: intercettare
un errore non è correggerlo, e $p'$ è la probabilità che il passo sia corretto
*oppure* che l'errore sia stato segnalato. Per leggerla come correttezza
efficace serve un'ipotesi in più, che conviene dichiarare: il passo
intercettato viene rifatto, e rifatto bene. È l'ipotesi giusta nella pratica,
perché la distinzione che conta è fra il fallimento **silenzioso**, che si
propaga travestito da dato di partenza, e quello **segnalato**, che è
recuperabile. Con $p = 0{,}95$ e un verificatore che ne pesca
l'$80\%$ si ottiene $p' = 0{,}99$, e su venti passi
$0{,}99^{20} = 0{,}818$ invece di $0{,}358$: da due catene su tre che
falliscono a una su cinque, cambiando **non il modello ma il ponteggio**.

Tre avvertenze per onestà. La prima è che $r$ **da solo non caratterizza un
verificatore**, e il modo più rapido di vederlo è portarlo all'estremo: con
$r = 1$ la formula dà $p' = 1$, cioè una catena lunga a piacere che non fallisce
mai qualunque sia la qualità dei passi. Ma $r = 1$ lo realizza anche il cancello
degenere che **rifiuta tutto**, il quale non verifica niente. Alla formula manca
il tasso di falso allarme, $P(\text{rifiuto} \mid \text{passo corretto})$: sotto
l'ipotesi appena dichiarata, che il passo intercettato venga rifatto, ogni falso
allarme è un passo rifatto per niente, ed è quel numero, non $r$, a dire quanto
costa il ponteggio. Un verificatore si prezza con due cifre, e la sezione «Come
falliscono» qui sotto guarda per lo più quella sbagliata, cioè il critico che
approva troppo, mentre il critico che boccia troppo è altrettanto reale e non
compare mai nel conto. La seconda avvertenza: il calcolo presuppone che il
controllo sia indipendente da chi produce (è il punto in cui un autocontrollo
non vale nulla, ci torniamo fra poco). La terza: i ritentativi dopo un rifiuto
non sono indipendenti fra loro, perché un modello che sbaglia tende a rifare lo
stesso ragionamento; contarli come prove indipendenti sovrastima il guadagno.

`````

Il meccanismo di controllo non va reinventato: è l'adulto col foglio in tasca del
gioco di poco fa, cioè un controllo **esterno** che dà sempre la stessa risposta
sullo stesso caso e non si lascia convincere da come gliela si racconta: un test
che gira, un conto che deve tornare, un formato che si valida. Nella sezione sul
loop engineering si chiama **validation gate**. Quello che i numeri qui sopra
aggiungono è la ragione per cui non è un lusso ma il pezzo portante: senza
cancello, ogni passo aggiunto alla catena è un fattore moltiplicativo minore
di uno; con il cancello, il fattore si avvicina a uno e la catena smette di
affondare.

## Quando si guadagna davvero

Fatti i tre conti, resta la domanda utile: esiste un caso in cui la squadra
vince? Sì, ne esistono tre, e conviene diffidare di chiunque ne elenchi un
quarto.

**Primo: il compito si decompone.** Se il lavoro si spezza in parti quasi
indipendenti (analizzare cinquanta documenti, provare otto ipotesi diverse,
tradurre venti capitoli) il parallelismo è reale e Amdahl è generoso, perché
$s$ è piccolo; e se aggiungendo agenti si analizzano *più* documenti invece che
gli stessi in meno tempo, il tetto di Amdahl non è nemmeno il metro giusto,
perché lì la taglia del problema non è fissa. Il segnale è preciso: le parti non devono scambiarsi
informazioni *durante* il lavoro, solo alla fine. Se invece i pezzi devono
consultarsi di continuo, non è decomposizione: è la trascrizione condivisa del
primo conto con un altro nome, e la bolletta è quella.

**Secondo: serve un giudizio indipendente da chi ha prodotto.** Qui il valore
aggiunto non è potenza di calcolo, è l'indipendenza: chi ha scritto il codice
ha già deciso, mentre lo scriveva, che è giusto. La separazione fra *maker* e
*checker* descritta nel loop engineering esiste per rompere questa
correlazione, e il guadagno si quantifica.

`````{tab} Elementare

È la differenza fra rileggersi il proprio tema e farlo rileggere a un compagno.
Rileggendoti, gli errori che non hai visto mentre scrivevi continui a non
vederli: sono proprio quelli su cui, scrivendo, avevi già deciso che andavano
bene. Un compagno che non sa che cosa avevi in testa li vede.

Mettiamoci dei numeri, scelti per fissare le idee e non misurati sul campo.
Diciamo che chi lavora sbaglia cinque volte su cento. Se poi si rilegge da solo,
di quei cinque errori se ne accorge grosso modo uno su dieci, e gli altri nove
arrivano in fondo: quattro volte e mezzo su cento. Se invece a rileggere è un
altro, che parte dal risultato e non conosce il ragionamento che c'è dietro, gli
sfugge un errore su tre invece di nove su dieci: in fondo ne arriva uno e mezzo
su cento, **tre volte meno**. Stesso modello, stessa bravura, stesso numero di
controlli: cambia solo chi controlla.

Qui sta la cosa da non sbagliare mai. Quel vantaggio non viene dall'avere due
teste, viene dal fatto che la seconda **non sa** come è stato fatto il lavoro. Se
al controllore si passa tutto il ragionamento del primo, la sua scrivania torna a
essere quella del primo, e il vantaggio sparisce insieme all'ignoranza.

`````

`````{tab} Superiore

Detta $P(E)$ la probabilità che il produttore sbagli e $P(M \mid E)$ quella che
il controllo non se ne accorga *dato che* l'errore c'è, un errore arriva in
fondo con probabilità $P(E)\,P(M \mid E)$. Poniamo $P(E) = 0{,}05$ e, per
fissare le idee, un autocontrollo che si accorge del proprio errore una volta su
dieci ($P(M \mid E) \approx 0{,}9$): lascia passare il $4{,}5\%$. Un critico con
contesto pulito e criteri propri scende a $P(M \mid E) \approx 0{,}3$ e lascia
passare l'$1{,}5\%$, tre volte meglio a parità di modello. Le due cifre sono
plausibili, non misurate; quello che conta è il verso della disuguaglianza, che
non dipende da quanto valgono esattamente. Tutto il valore del secondo agente
sta in quel condizionamento, e sparisce se gli si passa la trascrizione del
primo: a quel punto i due contesti tornano a coincidere.

`````

**Terzo: i contesti sono in conflitto.** È il caso più sottovalutato, e merita
di essere detto senza mezzi termini: qui il multi-agente non è un argomento di
intelligenza, è un argomento di **gestione del contesto**.

`````{tab} Elementare

Prova a fare due lavori diversi sulla stessa scrivania, con le carte di
entrambi mescolate. Non è che diventi meno intelligente: è che ogni volta che
cerchi un foglio ne trovi tre dell'altra pratica, e ogni tanto scrivi in un
documento una cosa che riguardava l'altro. Due scrivanie, una per pratica,
risolvono il problema senza che nessuno diventi più bravo.

Con gli agenti succede lo stesso, e per una ragione precisa: tutto quello che
finisce nella finestra di contesto (la scrivania del modello, cioè il testo
che si rilegge davanti prima di rispondere) continua a influenzare le risposte
successive, anche quello che si è rivelato sbagliato. Un'ipotesi tentata e
abbandonata al terzo turno resta scritta lì, e al ventesimo turno tira ancora
la risposta dalla sua parte. Dare a ogni agente una finestra pulita sul suo
pezzo non è un modo di avere più cervelli: è un modo di avere scrivanie
separate.

`````

`````{tab} Superiore

L'argomento si legge nel conto dei token, girato al contrario.
Con $N$ agenti a **contesti separati**, ciascuno con la propria finestra e
nessuna trascrizione condivisa (solo una sintesi che attraversa il confine
alla fine), il costo totale è $N$ volte quello di un agente singolo:

$$
\text{token}_{\text{separati}}(N, R)
\;=\; N\left[R\,c_0 + \bar{m}\,\frac{R(R+1)}{2}\right]
\;=\; O(N R^{2}),
$$

lineare in $N$ anziché quadratico. Con i numeri di prima ($N = 4$, $R = 8$,
$c_0 = 2000$, $\bar{m} = 500$) si passa da $328.000$ token a
$4 \times 34.000 = 136.000$: **quasi due volte e mezzo meno**, e il rapporto
rispetto al singolo agente torna a essere esattamente $4\times$, cioè quello
che l'intuizione si aspettava fin dall'inizio. In più la finestra massima
scende da $18.000$ a $6.000$ token, tre volte più stretta, il che sposta ogni
agente fuori dal regime in cui il *lost in the middle* morde.

Il punto architetturale generale è che **la topologia decide l'esponente**: la
stessa squadra di quattro agenti costa $O(N^2R^2)$ se tutti leggono tutto e
$O(NR^2)$ se ciascuno legge solo il proprio, e la differenza non è nel modello
ma in chi parla con chi. Il prezzo da pagare è che gli agenti sanno meno l'uno
dell'altro, e le informazioni che devono attraversare il confine vanno scelte
a mano: è il compromesso che la sezione sulle topologie tratta per esteso.

`````

## Come falliscono

Sapere che un sistema multi-agente può fallire non serve a niente; sapere *in
che modi* sì, perché ogni modo ha una contromisura diversa. Il lavoro di
riferimento è quello di Cemri e colleghi {cite}`cemri2025why`, che invece di
raccogliere aneddoti hanno fatto una cosa più noiosa e più utile. Hanno
raccolto le trascrizioni di sistemi multi-agente al lavoro davvero, cioè tutto
quello che gli agenti si erano detti e avevano fatto dall'inizio alla fine, su
sette dei framework più diffusi. Sei esperti ne hanno lette e catalogate a
mano centocinquanta, prese da cinque di quei framework, mettendosi prima
d'accordo su come giudicarle e
poi misurando quanto spesso, sulla stessa trascrizione, davano davvero lo
stesso giudizio; da lì è uscito un elenco ordinato dei modi di fallire, che
chiamano MAST. Con l'elenco in mano la catalogazione è stata poi estesa a
milleseicentoquarantadue trascrizioni in tutto, affidandola a un modello al
posto delle persone dopo aver controllato che sui casi già giudicati dagli
esperti desse le stesse risposte.
Ne escono quattordici modi ricorrenti in tre famiglie, e
sono le famiglie la parte da ricordare, perché dicono *dove* guardare.

La prima è quella delle **specifiche e del progetto del sistema**. L'agente
esegue alla lettera un compito sottospecificato, o esce dal ruolo assegnato, o
ripete un passo già fatto, o non riconosce la condizione in cui deve fermarsi.
Il tratto comune è che il difetto non sta nel modello: sta in ciò che gli è
stato scritto (o non scritto) all'inizio. Un compito enunciato male produce
un'esecuzione impeccabile della cosa sbagliata, ed è il modo di fallire più
frequente e meno spettacolare.

La seconda è il **disallineamento fra agenti**. Ognuno ha una sua versione
dello stato del mondo, le versioni divergono, e nessuno se ne accorge perché
la conversazione resta perfettamente fluente. Qui stanno l'agente che ignora
quello che gli è stato appena detto, quello che procede su una richiesta
ambigua senza chiedere chiarimenti, quello che trattiene un'informazione utile
agli altri, e la conversazione che scivola su un compito diverso senza che
nessuno lo dichiari. È la famiglia più insidiosa perché non produce sintomi
leggibili: due agenti in disaccordo su cosa stanno facendo si scrivono
messaggi cortesi e ben formati fino in fondo.

La terza è la **verifica e la terminazione**. Il sistema si ferma prima di
aver finito, oppure la verifica manca del tutto, oppure c'è ma è inadeguata:
il critico approva perché non ha modo di provare davvero, e il suo verdetto non
aggiunge informazione, aggiunge una firma. Torna qui, in forma empirica, la
matematica di poche righe fa: un controllo che intercetta quasi nessuno degli
errori lascia la catena esattamente dov'era, e il ponteggio è teatro.

Le tre famiglie di MAST hanno tre contromisure diverse, e nessuna delle tre è
«usare un modello più bravo»: si
scrivono specifiche più strette, si rende esplicito lo stato condiviso invece
di lasciarlo implicito nella conversazione (è il mestiere dei protocolli e dei
meccanismi di consenso, più avanti in questo capitolo), si dà al verificatore
un criterio che possa davvero applicare.

## La regola prudente

Il capitolo sugli Agenti chiudeva con una formula che vale la pena rendere
verificabile: si aggiunge un ruolo solo quando risolve un problema reale, cioè
un problema che un agente da solo non risolveva. La versione operativa ha
quattro clausole.

**Il problema deve essere documentato, non intuito.** Prima di aggiungere il
critico bisogna poter esibire i casi in cui il solista sbagliava e il critico
avrebbe intercettato. Se quei casi non si trovano nelle tracce, il ruolo sta
risolvendo un problema immaginario.

**Il confronto è contro un singolo agente ben progettato.** È la clausola che
salta più spesso, e senza di essa qualunque architettura vince. Un solista con
un buon prompt di sistema, gli strumenti giusti e un validation gate è un
avversario ben diverso da un solista improvvisato, ed è contro di lui che il
conto va fatto. La letteratura, richiamata proprio in apertura del lavoro sui
fallimenti reali, riporta che sui banchi di prova diffusi il guadagno delle
squadre resta spesso minimo rispetto ai sistemi a un agente solo, e perfino
rispetto a confronti elementari come campionare $N$ risposte dallo stesso
modello e tenere la migliore secondo un verificatore {cite}`cemri2025why`; ed è
una letteratura giovane, ancora lontana da un verdetto netto {cite}`xi2023rise`.

**A parità di budget.** Se la squadra consuma 328.000 token, il termine di
paragone onesto non è il solista da 34.000: è il solista a cui si danno gli
stessi 328.000, spesi in campionamenti ripetuti con voto di maggioranza, in
ragionamento più lungo, in più tentativi contro il cancello di verifica. Molti
guadagni attribuiti al multi-agente sono, misurati così, guadagni attribuibili
al calcolo aggiuntivo, e si sarebbero ottenuti anche senza squadra.

**Il ruolo si tiene solo se una misura lo conferma.** La prova è l'ablazione:
si toglie il ruolo, si rimisura, e se il numero non si muove il ruolo esce. Le
misure sono quelle già viste per gli agenti (tasso di successo, qualità della
traiettoria, costo e latenza), lette insieme e mai una sola: un'architettura
che alza il successo di due punti e il costo di dieci volte non ha vinto, ha
speso.

Sotto tutte e quattro c'è la lezione di Brooks mezzo secolo dopo: il
coordinamento non è gratis e non è neutro, è una voce di costo che cresce più
in fretta del beneficio che finanzia. Un sistema multi-agente ben progettato
non è quello con più agenti: è quello con il minimo numero di agenti che
risolve il problema, ciascuno con una finestra pulita e un compito che il
solista non chiudeva.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Con una **trascrizione condivisa** (la chat di gruppo in cui, prima di
  scrivere, si rilegge tutto dall'inizio) raddoppiare i partecipanti non
  raddoppia il costo: quasi lo quadruplica, perché ogni messaggio nuovo dovrà
  essere riletto da tutti quelli che parleranno dopo. Quattro agenti per otto
  giri leggono $328.000$ token contro i $34.000$ di un solista: **quasi dieci
  volte**, non quattro.
- La parte di lavoro che **non si può dividere** mette un tetto (è la **legge di
  Amdahl**, il tetto di questa sezione). Se su dieci ore di rapporto tre
  servono a decidere la scaletta e a rileggere alla fine, nessun numero di
  collaboratori chiude quel rapporto in meno di tre ore: oltre tre volte più
  veloci non si va. E siccome ogni collaboratore in più fa perdere
  tempo anche agli altri, la curva a un certo punto non si appiattisce soltanto,
  torna giù: esiste un numero migliore di agenti, e oltre quello si peggiora.
- Gli errori si **moltiplicano**, non si mediano: è il telefono senza fili. Con
  partecipanti che riferiscono bene nel $95\%$ dei casi, dieci passaggi lasciano
  la frase intatta sei volte su dieci e venti passaggi poco più di tre. Basta
  però un controllo esterno (il **validation gate**) che intercetti l'$80\%$
  degli errori e faccia rifare il passaggio, e i venti passaggi tornano a
  riuscire otto volte su dieci: la verifica non è un lusso, è ciò che spezza la
  catena.
- Si guadagna davvero in **tre casi soli**: il compito si spezza in parti che
  non hanno bisogno di parlarsi mentre lavorano; serve un **giudizio
  indipendente** da chi ha prodotto (il valore sta nel non aver già deciso, non
  nella potenza aggiunta); due lavori si **disturbano a vicenda** e conviene
  dare a ciascuno la sua scrivania, cioè un contesto pulito, e allora il conto
  cresce in proporzione agli agenti invece che al loro quadrato.
- I fallimenti reali {cite}`cemri2025why` si raggruppano in tre famiglie:
  **specifiche e progetto** (esecuzione impeccabile di un compito detto male),
  **disallineamento fra agenti** (ognuno ha una versione diversa di che cosa sta
  succedendo, sotto una conversazione perfettamente fluente), **verifica e
  terminazione** (si approva senza poter davvero controllare, o ci si ferma
  troppo presto).
- La **regola prudente**: il problema va documentato e non intuito, il confronto
  si fa contro un **singolo agente ben progettato** e **a parità di token
  spesi**, e il ruolo si tiene solo se togliendolo il risultato peggiora davvero
  {cite}`xi2023rise`.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Con una **trascrizione condivisa** il costo va come il quadrato dei turni,
  $\sum_{t=1}^{T}(c_0 + t\bar{m}) = O(\bar{m}T^2)$, e siccome $T = NR$ è
  quadratico anche negli agenti. Quattro agenti per otto giri leggono
  $328.000$ token contro i $34.000$ di un solista: **quasi dieci volte**, non
  quattro.
- La **legge di Amdahl** mette un tetto: con una frazione seriale $s$
  l'accelerazione è al più $1/(s + (1-s)/N)$, cioè $1/s$ per $N \to \infty$.
  Con $s = 0{,}3$ nessun numero di agenti supera $3{,}3\times$; e con una
  penale di coordinamento la curva torna giù dopo $N^{*}=\sqrt{(1-s)/\kappa}$.
- Gli errori si **compongono**: $n$ passi corretti con probabilità $p$ danno
  $p^n$. Con $p = 0{,}95$, dieci passi danno $0{,}60$ e venti $0{,}36$. Un
  **validation gate** che intercetta l'$80\%$ degli errori (e fa rifare il
  passo intercettato) porta $p'$ a $0{,}99$ e i venti passi a $0{,}82$: la
  verifica non è un lusso, è ciò che spezza la catena moltiplicativa.
- Si guadagna davvero in **tre casi soli**: il compito si decompone in parti
  quasi indipendenti; serve un **giudizio indipendente** (il valore è nella
  decorrelazione, non nella potenza aggiunta); i **contesti sono in conflitto**
  e conviene separarli, e allora il multi-agente è gestione del contesto (costo
  $O(NR^2)$ invece di $O(N^2R^2)$: la topologia decide l'esponente).
- I fallimenti reali {cite}`cemri2025why` si raggruppano in tre famiglie:
  **specifiche e progetto** (esecuzione impeccabile di un compito
  sottospecificato), **disallineamento fra agenti** (stati del mondo divergenti
  sotto una conversazione fluente), **verifica e terminazione** (si approva
  senza poter provare, o ci si ferma troppo presto).
- La **regola prudente**: problema documentato, confronto contro un **singolo
  agente ben progettato** e **a parità di budget di token**, e il ruolo si
  tiene solo se l'ablazione lo conferma {cite}`xi2023rise`.
```

`````
