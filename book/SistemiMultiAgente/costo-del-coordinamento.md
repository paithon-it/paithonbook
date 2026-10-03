# Il costo del coordinamento

Nel 1975 Fred Brooks pubblica *The Mythical Man-Month*, il resoconto di cosa
fosse andato storto nella costruzione dell'OS/360 alla IBM. La tesi che lo ha
reso celebre è controintuitiva: aggiungere programmatori a un progetto software
già in ritardo lo fa ritardare di più. Le ragioni che porta sono tre. I nuovi
arrivati vanno istruiti da chi già lavora, e il tempo per istruirli lo toglie
proprio chi è più avanti. Una parte del lavoro non si divide, per quante persone
ci si mettano. E i canali di comunicazione da tenere aperti crescono più in
fretta delle persone: due persone ne hanno uno, tre ne hanno tre, quattro sei,
dieci quarantacinque, cioè $n(n-1)/2$ per $n$ persone. Il lavoro utile, intanto,
al massimo raddoppia quando raddoppiano le persone; i canali no. Passata una
certa soglia il tempo speso a tenersi aggiornati si mangia il tempo guadagnato
lavorando in più.

Chi mette al lavoro più agenti si trova davanti un conto della stessa forma, ma
di origine diversa. Qui a moltiplicarsi non sono i canali fra le persone ma la
conversazione, che si allunga a ogni intervento e che ogni agente rilegge da
capo prima di parlare. Quello che si rilegge sono token, cioè i frammenti in
cui un modello di linguaggio scompone il testo prima di elaborarlo: in italiano
un token vale grosso modo mezza parola (il rapporto esatto dipende da come il
modello spezza le parole), e una pagina di quattrocento parole ne contiene
sette o ottocento. I token sono anche l'unità in cui si paga, perché chi mette
a disposizione il modello li conta e li fattura, e li conta due volte: quelli
che il modello scrive e quelli che gli si fanno leggere. La {doc}`sezione sulle
architetture degli agenti </Agenti/architetture-e-valutazione>` ha già detto, a
parole, che una squadra costa più di un agente che lavora da solo (un
*solista*, come lo chiameremo) e moltiplica i modi di sbagliare. Qui
quell'avvertimento diventa un conto: tre formule che si fanno su un tovagliolo
prima di scrivere una riga di codice.

Le stime di questo genere, fatte a spanne prima di avere i dati, portano il
nome di Enrico Fermi. Il 16 luglio 1945, al test Trinity, cioè la prima
esplosione di una bomba atomica, lasciò cadere dei pezzetti di carta al
passaggio dell'onda d'urto, guardò di quanto si fossero spostati e stimò a
mente la potenza: una decina di chilotoni, cioè quanto diecimila tonnellate di
tritolo. La cifra stabilita in seguito è circa il doppio (ventuno chilotoni, e
una rianalisi dei dati pubblicata nel 2021 la porta a quasi venticinque), il
che sembra un errore grosso finché non si pensa all'alternativa: nessun numero
affatto, fino a quando i dati non fossero stati raccolti e analizzati. Per
decidere se una cosa è ragionevole, sapere subito che si è dalle parti delle
diecimila tonnellate e non delle dieci vale più che aspettare mesi per la cifra
esatta. Prima di assemblare una squadra conviene fare lo stesso: quanto
costerà, quanto potrà accelerare, quanto spesso sbaglierà.

## Il conto dei token

Lo schema più semplice, quello della conversazione di gruppo di AutoGen
{cite}`wu2024autogen`, è la **trascrizione condivisa**: gli agenti scrivono
tutti nello stesso verbale, cioè il resoconto scritto della riunione, e
ognuno, quando tocca a lui, lo rilegge dall'inizio prima di rispondere. È la
scelta più naturale (nessuno resta indietro, tutti vedono tutto) ed è anche
quella in cui il conto cresce più in fretta del numero di interventi, per una
ragione che riguarda la lunghezza della trascrizione e non l'intelligenza dei
partecipanti.

`````{tab} Elementare

In una chat di gruppo vale una regola: prima di scrivere un messaggio devi
rileggere *tutta* la conversazione dall’inizio. Il primo che parla legge un
messaggio (l'enunciato del compito). Il secondo ne legge due. Il decimo ne
legge dieci. Il trentaduesimo ne legge trentadue. Nessuno ha scritto messaggi
più lunghi degli altri, eppure la fatica totale di lettura non cresce come il
numero di interventi: cresce molto più in fretta, perché ogni messaggio nuovo
dovrà essere riletto da tutti quelli che parleranno dopo.

Contiamo. Con otto interventi in tutto, i messaggi letti sono uno più due più
tre e così via fino a otto, e il modo rapido di sommarli è accoppiare il primo
con l'ultimo: uno più otto fa nove, due più sette fa nove, e di coppie da nove
ce ne sono quattro: 36. Con trentadue interventi le coppie diventano sedici e
ciascuna vale trentatré: 528.

Quattro volte gli interventi, quasi quindici volte le righe lette. E il
quindici non arriva tutto insieme: a sedici interventi le righe lette sono 136,
quasi quattro volte 36, e da 136 a 528 è di nuovo quasi quattro volte. Ecco
tutto il problema: a ogni raddoppio degli interventi le righe da leggere si
quadruplicano.

Sul conto vero il moltiplicatore scende da quindici a dieci, e la ragione è un
cartello. In cima alla chat c'è il regolamento, con scritto chi fa che cosa, che
ciascuno rilegge a ogni turno e che non si allunga mai: una parte ferma,
sommata a una che si gonfia, fa crescere il totale più lentamente della parte
che si gonfia. Mettiamo che il cartello sia lungo quanto quattro messaggi. Con
otto interventi si leggono otto cartelli, cioè 32 messaggi, più i 36 di prima:
68. Con trentadue interventi i cartelli fanno 128 messaggi, più i 528: 656. E
656 diviso 68 fa quasi dieci, che è il numero da ricordare.

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
sui turni, non sugli agenti; un sistema con due agenti molto loquaci può
costare più di uno con otto agenti che dicono una frase a testa. La seconda:
poiché $T$ è proporzionale a $N$, il costo *è* comunque quadratico nel numero
di agenti. Cresce come i canali $n(n-1)/2$ di Brooks, ma per un'altra ragione:
qui il quadrato non viene dalle coppie che si parlano, viene dalla rilettura
cumulativa di una trascrizione che si allunga, ed è quadratico nei turni
anche per un solista che lavori da solo abbastanza a lungo. Stessa forma,
meccanismo diverso. (Qui la finestra del primo turno contiene già l'enunciato
del compito; nel conto a parole della sezione sulle architetture degli agenti il
primo intervento non rilegge niente, $\sum_{t=0}^{T-1} t$ invece di
$\sum_{t=1}^{T} t$, e le due somme differiscono di $T$ messaggi.)

Il conto è quello dei token in ingresso, e due precisazioni lo collocano. La
prima riguarda il prezzo. I fornitori riusano i calcoli fatti su un prefisso
già visto e lo fatturano a prezzo ridotto, il *prompt caching* della
{doc}`sezione sul contesto </IngegneriaLLM/context-engineering>`; in una
trascrizione condivisa il prefisso di ogni richiesta è quasi tutto già passato
da quell'agente al turno precedente, e a prezzo pieno restano soprattutto i
messaggi arrivati nel frattempo. Lo sconto abbassa la costante, non
l'esponente: i token letti restano $O(\bar{m}\,T^2)$, e non cambiano né la
larghezza della finestra né l'effetto *lost in the middle*. La seconda riguarda
il calcolo, quando il prefisso non viene riusato: con attenzione piena,
elaborare un contesto di lunghezza $L$ ha un termine quadratico in $L$, e
siccome qui $L$ cresce linearmente con i turni, sommato sulla conversazione quel
termine va come $T^{3}$.

L'altro regime possibile è il *broadcast*, dove a ogni giro tutti leggono e
tutti scrivono insieme invece che a turno. Ciascuno legge
l'enunciato del compito e la trascrizione dei giri già chiusi, che cresce di
$N\bar{m}$ a giro, per un totale di

$$
\text{token}_{\text{broadcast}}(N, R) \;=\; N R\,(c_0 + \bar{m}) \;+\;
N^{2}\,\bar{m}\,\frac{R(R-1)}{2},
$$

cioè lo stesso ordine $O(\bar{m}N^2R^2)$ con la stessa costante di testa:
lo sconto sta in un termine di grado inferiore, e quindi si vede sui numeri
piccoli e sparisce al crescere dei turni. Con quattro agenti e otto giri a testa
($N = 4$, $R = 8$) fa $304.000$ token contro i
$328.000$ del turno a turno, il 7% in meno; a $R = 64$ il rapporto è già
$0{,}99$. Resta sotto uno per
ogni squadra di almeno due agenti, ma sempre meno. Attenzione a non leggerlo
come un risparmio: la ragione dello
sconto è che dentro un giro nessuno legge gli altri. Chi parla per $t$-esimo
nel regime a turni ha già davanti i $t-1$ interventi del giro in corso; in
*broadcast* tutti leggono la stessa trascrizione, ferma al giro precedente, e
quello che manca dal conto è esattamente l'informazione che manca a chi lavora.
Ciò che il *broadcast* compra davvero è la latenza, $R$ giri invece di $NR$
turni, perché gli interventi di un giro si producono in parallelo; e la compra
al prezzo di un giro di ritardo nel reagire a quello che ha appena detto un
altro.

`````

Mettiamo delle taglie tonde: duemila token di istruzioni iniziali (due o tre
pagine di regole, che ciascuno si rilegge sempre uguali a ogni turno), messaggi
da cinquecento token l'uno (mezza paginetta), otto giri di parola a testa. Poche
righe di Python sommano i token letti e calcolano anche la larghezza della
finestra di contesto all'ultimo turno, cioè il testo che ha davanti chi parla
per ultimo:

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
otto giri leggono 328.000 token contro i 34.000 di un solista che lavora
otto turni: non quattro volte tanto, quasi dieci volte tanto. I token
*scritti*, invece, sono esattamente quattro volte (32 turni per 500 token fanno
16.000, contro gli 8 per 500 del solista, cioè 4.000): tutto
lo scarto sta sul lato della lettura, la parte del conto che nessuno guarda
perché non produce niente di visibile.

C'è un secondo effetto, che nel conto non compare ed è più insidioso.
All'ultimo turno la finestra di chi parla contiene 18.000 token di
conversazione già avvenuta, e l'informazione decisiva (una decisione presa al
decimo turno, un vincolo enunciato al terzo) sta sepolta lì in mezzo. È la
posizione peggiore per l'effetto *lost in the middle* {cite}`liu2024lost`, già
incontrato nel {doc}`context engineering </IngegneriaLLM/context-engineering>`:
su compiti di recupero e di domande su più documenti l'accuratezza è più alta
quando l'informazione sta all'inizio o alla fine del contesto e scende quando
sta nel mezzo, in misura diversa da un modello all'altro. Una trascrizione
condivisa costa cara, ed è anche il posto peggiore dove mettere qualcosa che
deve essere ricordato.

## Il tetto di Amdahl

Il costo è metà del conto. L'altra metà è il guadagno, e anche lì c'è una
legge che non si aggira. Nel 1967 Gene Amdahl, l'architetto del System/360 di
IBM, portò a un convegno tre pagine scritte per raffreddare gli entusiasmi
verso i calcolatori a più processori: una parte del lavoro, sosteneva, resta
sequenziale comunque, e quella parte da sola basta a mettere un tetto.
L'argomento era in prosa; la formula che oggi porta il suo nome è stata
ricavata da quelle pagine dai commentatori venuti dopo, e vale per qualunque
cosa si provi a spartire fra più mani, che siano processori, persone o agenti.
La parte di lavoro che *non si può* spezzare mette un tetto a quanto si può
andare più veloci, e quel tetto non si alza aggiungendo mani. In una squadra di
agenti la parte indivisibile è facile da riconoscere: qualcuno deve decidere il
piano prima che gli altri comincino, qualcuno deve ricucire i pezzi dopo che
hanno finito. Piano e sintesi sono la strozzatura all'ingresso e all'uscita.

`````{tab} Elementare

Un rapporto da consegnare costa dieci ore di lavoro. Tre di quelle ore non si
dividono con nessuno: un'ora per decidere la scaletta (finché non c'è, nessuno
può scrivere il proprio capitolo) e due ore alla fine per rileggere tutto
insieme e uniformare il tono. Le altre sette sono i capitoli, e quelle sì che
si spartiscono. Il rapporto, si intende, resta sempre quello: non è che
assumendo gente lo si allunga.

Con quattro persone le sette ore di scrittura diventano un'ora e tre quarti, e
il lavoro dura tre ore più un'ora e tre quarti: quattro ore e tre quarti.
Quante volte si è andati più veloci lo dice dieci diviso quattro e tre quarti:
poco più di due volte, non quattro.

Con otto persone la scrittura scende a poco meno di un'ora e il lavoro dura
tre ore e cinquantadue minuti e mezzo: poco più di due volte e mezzo. I
quattro assunti in più hanno comprato mezza volta di velocità; i primi quattro
ne avevano comprata una intera. E per quanta gente si assuma, le tre ore di
scaletta e rilettura restano lì: sotto le tre ore quel rapporto non scende,
cioè più di tre volte e un terzo non si va nemmeno con un milione di
persone.

A quel tetto, poi, non ci si arriva, perché fin qui chi arriva non costava
niente a chi c'era già. Invece va messo al corrente, fa domande, aspetta
risposte: mettiamo dodici minuti tolti al gruppo per ogni arrivo. Allora i
quattro chiudono in cinque ore e ventuno, gli otto in cinque ore e sedici, e i
sedici in sei ore e ventisei, cioè peggio dei quattro. Molto prima del tetto c'è
un numero di collaboratori oltre il quale assumere allunga il lavoro invece di
accorciarlo: qui è sei, e le sei chiudono in cinque ore e dieci.

`````

`````{tab} Superiore

Sia $s \in [0,1]$ la frazione intrinsecamente seriale del lavoro (quella
che va svolta da un solo esecutore, indipendentemente da quanti ce ne siano) e
$1-s$ la frazione parallelizzabile. L'ipotesi nascosta, e conviene dirla perché
è quella che si viola più spesso, è che il problema abbia taglia fissa: si
divide sempre lo stesso lavoro fra più esecutori, non se ne fa di più.
Normalizzando a 1 il tempo del singolo
esecutore, con $N$ esecutori il tempo è $s + (1-s)/N$ e l'accelerazione vale

$$
S(N) \;=\; \frac{1}{\,s + \dfrac{1-s}{N}\,},
\qquad
\lim_{N \to \infty} S(N) \;=\; \frac{1}{s},
$$

dove $S(N)$ è il rapporto fra il tempo con un esecutore e il tempo con $N$. Il
limite è la **legge di Amdahl** {cite}`amdahl1967validity`, la stessa del
{doc}`capitolo sulle GPU </GPU/overview>`, dove compare con la frazione
parallela $p = 1-s$ al posto di quella seriale: il tetto è $1/s$ e non dipende
da $N$. Con $s = 0{,}3$ nessun numero di agenti porta oltre $3{,}33\times$, e la
salita verso il tetto è di quelle che si spengono in fretta:

| $N$ | 1 | 2 | 4 | 8 | 16 | $\infty$ |
|---|---|---|---|---|---|---|
| $S(N)$ | $1{,}00$ | $1{,}54$ | $2{,}11$ | $2{,}58$ | $2{,}91$ | $3{,}33$ |

Per arrivare al $90\%$ del tetto servono $N = 21$ agenti; per arrivare al
tetto, infiniti. Ma questa curva è un limite superiore ottimistico, perché
assume che coordinare non costi nulla, e il conto dei token ha
appena mostrato che costa. Se si aggiunge una penale di coordinamento lineare,
cioè un tempo $\kappa(N-1)$ che ogni agente in più impone a tutti gli altri, il
tempo diventa $s + (1-s)/N + \kappa(N-1)$ e la curva non si limita ad
appiattirsi: torna giù, e l'ottimo si trova annullando la derivata,

$$
N^{*} \;=\; \sqrt{\frac{1-s}{\kappa}},
$$

dove $\kappa$ è il costo di sincronizzazione per agente aggiunto, espresso
nella stessa unità del tempo totale. Con $s = 0{,}3$ e $\kappa = 0{,}02$ (ogni
agente in più aggiunge il $2\%$ del tempo originario in coordinamento) si
ottiene $N^{*} = \sqrt{35} \approx 5{,}9$: l'ottimo è a sei agenti,
l'accelerazione massima è $1{,}94\times$ (contro un tetto di Amdahl di
$3{,}33$), e a sedici agenti si è già scesi a $1{,}55\times$, peggio che con
quattro. Il ginocchio della curva, non il tetto, è il numero che conta. E la
penale lineare è l'ipotesi generosa: se il coordinamento cresce come il
quadrato degli agenti, come suggerisce il conto dei token, il
ginocchio si sposta ancora più a sinistra.

`````

La conseguenza pratica è una sola: si aggiungono agenti finché il pezzo
divisibile lo giustifica, e poi si smette. La domanda da farsi non è «un altro
agente aiuterebbe?» (la risposta è quasi sempre sì, di pochissimo) ma «quanto
vale l’*ultimo* agente aggiunto, rispetto a quello che costa?».

I due conti si possono mettere accanto, se li si fa sulla stessa squadra. Con
la frazione seriale del rapporto, tre ore su dieci, passare da quattro a otto
agenti porta l'accelerazione da 2,11 a 2,58 volte, cioè la aumenta del 22%;
nella trascrizione condivisa del conto dei token, intanto, i token letti
passano da 328.000 a 1.168.000, 3,6 volte tanto. E quel 22% è ancora la stima
generosa, perché è il guadagno che si otterrebbe se coordinarsi non costasse
niente: contando anche il tempo che ogni nuovo arrivato fa perdere agli altri,
gli otto non guadagnano quasi più nulla rispetto ai quattro.

Il caso opposto esiste, ed è quello in cui i partecipanti sono centinaia,
elementari, e coordinarli non costa quasi niente: sono gli sciami, trattati
nella {doc}`sezione su sciami e simulazioni
</SistemiMultiAgente/sciami-e-simulazioni>`.

## Gli errori si compongono

Il terzo conto è quello che manda a picco le catene lunghe di agenti, ed è il
più semplice dei tre. Se un lavoro passa di mano in mano, e ogni passaggio
riesce o fallisce per conto proprio, quanto è affidabile l'insieme sta molto
sotto la media di quanto sono affidabili i passaggi, perché per arrivare in
fondo devono riuscire tutti.

`````{tab} Elementare

È il gioco del telefono senza fili. Ogni bambino ripete alla persona accanto
quello che ha sentito, e supponiamo che sia bravissimo: novantacinque volte su
cento ripete la frase esatta. Sembra una squadra affidabilissima.

Perché la frase arrivi intatta in fondo devono andare bene *tutti* i passaggi,
uno dopo l'altro. Con due passaggi: il primo riesce 95 volte su 100, e di quelle
95 il secondo ne rovina 5 su 100, quindi ne restano poco più di 90. Ogni
passaggio toglie la sua parte da quello che è arrivato fin lì, e allora bisogna
moltiplicare fra loro tanti novantacinque-su-cento quanti sono i passaggi. Con
dieci passaggi si scende a sessanta volte su cento; con venti a trentasei, cioè
poco più di tre volte su dieci. Nessuno è diventato meno bravo: è la fila che si
è allungata.

Girato dall'altro verso, il conto scoraggia. Perché una fila di venti passaggi
riesca nove volte su dieci, ogni bambino dovrebbe sbagliare meno di sei volte
su mille: è difficilissimo arrivarci, sui compiti in cui non c'è una risposta
sola già prevista.

Ora però mettiamo un arbitro a ogni passaggio: uno che, senza sapere qual è
la frase giusta, sa dire se quella che ha appena sentito sta in piedi, e in caso
contrario fa ripetere. Non deve nemmeno essere infallibile. Diciamo che di dieci
parole cambiate se ne accorge otto: la fila di venti passaggi torna a riuscire
otto volte su dieci, invece di tre.

Anche un arbitro solo, messo a metà fila, recupera una parte del guadagno,
cioè della distanza fra le tre volte su dieci della fila senza arbitri e le otto
della fila con un arbitro a ogni passaggio. Dopo dieci passaggi la frase è
rovinata quattro volte su dieci; l'arbitro se ne accorge in otto casi su dieci
e fa rifare la prima metà, e supponiamo che la seconda volta venga bene.
Così la frase rovinata che passa oltre scende a meno di una volta su dieci. La
seconda metà la rovina ancora quattro volte su dieci, e in fondo la fila
riesce poco più di cinque volte su dieci invece di tre: poco più di quattro
decimi del guadagno, con un arbitro solo. Se invece anche la prima metà rifatta
riesce solo sei volte su dieci, come la prima volta, si scende a meno di cinque:
un arbitro solo recupera allora circa un quarto del guadagno.

Due cose, però, l'arbitro non le aggiusta. Uno che dicesse «ripeti» a chiunque
non lascerebbe passare nessun errore e non farebbe arrivare in fondo neanche una
frase giusta: quanto vale un arbitro si vede anche da quante volte ferma chi
aveva ragione. E il bambino a cui fai ripetere spesso ripete lo stesso sbaglio,
perché la parola l'aveva sentita male fin dall'inizio; il guadagno vero è un po’
meno di quello che il conto promette.

`````

`````{tab} Superiore

Sia $p$ la probabilità che un singolo passo sia corretto, e siano gli $n$
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
mille passi: un requisito che si può garantire su operazioni semplici e ben
specificate, e che è difficile garantire sui passi di un compito aperto, dove
ogni passo è una decisione. Questa, più che l'incapacità dei singoli agenti, è
la ragione strutturale per cui le pipeline lunghe falliscono: l'affidabilità non
si somma, si moltiplica.

La stessa formula dice però anche come uscirne senza cambiare modello. Se un
controllo intercetta una frazione $r$ degli errori (il *recall* del
verificatore) e il passo intercettato, rifatto, riesce con probabilità $q$, la
probabilità che il passo sia corretto alla fine sale a

$$
p' \;=\; p + (1-p)\,r\,q,
$$

dove $1-p$ è la probabilità che il passo sbagli, $r$ quella che il controllo se
ne accorga e $q$ quella che il rifacimento vada bene. Attenzione a che cosa
misura $p'$: intercettare un errore non è correggerlo, e la formula conta come
recuperato soltanto l'errore intercettato *e* rifatto bene. Con $q = 1$ si
ritrova la forma che si incontra più spesso, $p' = 1 - (1-p)(1-r)$, che però
letta da sola conta come buono anche l'errore soltanto segnalato. La
distinzione che conta, nella pratica, è fra il fallimento silenzioso, che si
propaga travestito da dato di partenza, e quello segnalato, che è recuperabile.
Con $p = 0{,}95$, un verificatore che ne pesca l’$80\%$ e rifacimenti sempre
riusciti si ottiene $p' = 0{,}99$, e su venti passi $0{,}99^{20} = 0{,}818$
invece di $0{,}358$: da due catene su tre che falliscono a una su cinque,
cambiando non il modello ma il ponteggio. Se però il rifacimento riesce
soltanto sei volte su dieci, $q = 0{,}6$, si ha $p' = 0{,}974$ e
$p'^{20} = 0{,}59$: il verificatore recupera ancora molto, ma circa la metà del
salto promesso dalla forma con $q = 1$.

Tre avvertenze per onestà. La prima è che $r$ da solo non caratterizza un
verificatore, e il modo più rapido di vederlo è portarlo all'estremo: con
$r = 1$ e $q = 1$ la formula dà $p' = 1$, cioè una catena lunga a piacere che
non fallisce mai qualunque sia la qualità dei passi. Ma $r = 1$ lo realizza
anche il cancello degenere che rifiuta tutto, il quale non verifica niente. Alla
formula manca il tasso di falso allarme,
$P(\text{rifiuto} \mid \text{passo corretto})$: sotto l'ipotesi appena
dichiarata, che il passo intercettato venga rifatto, ogni falso allarme è un
passo rifatto per niente, ed è quel numero, non $r$, a dire quanto costa il
ponteggio. Un verificatore si prezza con due cifre, e i cataloghi dei fallimenti
reali guardano per lo più quella sbagliata, cioè il critico che approva troppo,
mentre il critico che boccia troppo è altrettanto reale e non compare mai nel
conto. La seconda avvertenza: il calcolo presuppone che il controllo sia
indipendente da chi produce, ed è il punto in cui un autocontrollo non vale
nulla. La terza: i ritentativi dopo un rifiuto non sono indipendenti fra loro,
perché un modello che sbaglia tende a rifare lo stesso ragionamento; contarli
come prove indipendenti sovrastima il guadagno.

`````

Il verificatore non va reinventato. È un controllo esterno, che dà sempre la
stessa risposta sullo stesso caso e non si lascia convincere da come gliela si
racconta. E non gli serve conoscere la risposta giusta: gli basta saper
riconoscere una risposta che non sta in piedi. Sono cose come un programma di
prova che si esegue e o passa o non passa, un conto che deve tornare, un modulo
che deve avere tutte le caselle riempite. Nella sezione sul {doc}`loop
engineering </IngegneriaLLM/loop-engineering>` quel controllo si chiama cancello
di verifica (in inglese *validation gate*), e il nome dice il mestiere: chi non
è in regola non passa.

I conti sul costo del coordinamento aggiungono la ragione per cui il cancello
non è un lusso ma il pezzo portante. Senza, ogni passaggio in più porta via la
stessa frazione di quello che è arrivato fin lì, e le perdite si compongono:
dopo $n$ passaggi resta $p^n$, che scende verso zero senza mai fermarsi. Con il
cancello la frazione persa a ogni passaggio diventa piccola, e la fila affonda
molto più piano man mano che si allunga.

## Quando si guadagna davvero

Fatti i tre conti, resta la domanda utile: esiste un caso in cui la squadra
vince? Ne esistono almeno tre che i conti appena fatti sanno valutare. Un quarto
lo ha già mostrato la squadra dell'apertura del capitolo, dove soltanto il
coordinatore può eseguire codice sulla macchina vera: separare i permessi fra
agenti è una ragione di sicurezza per dividere il lavoro, che nessuno dei tre
conti misura.

Primo: il compito si decompone. Se il lavoro si spezza in parti quasi
indipendenti (analizzare cinquanta documenti, provare otto ipotesi diverse,
tradurre venti capitoli) allora la parte che non si può dividere è piccolissima,
e il tetto di poco fa sta molto in alto. C'è anche il caso migliore di tutti: se
aggiungendo agenti si analizzano *più* documenti invece degli stessi in meno
tempo, il metro giusto non è più la legge di Amdahl, che suppone un lavoro di
taglia fissa, ma quella di Gustafson {cite}`gustafson1988reevaluating`, detta
della *scalabilità debole*: se la parte divisibile cresce con le mani e quella
seriale no, con una frazione seriale $s$ misurata sul tempo del lavoro
parallelo l'accelerazione vale $s + (1-s)N$, e non ha tetto. Il segnale che
distingue i due casi è preciso: le parti non devono scambiarsi informazioni
*durante* il lavoro, solo alla fine. Se invece i pezzi devono consultarsi di
continuo, quella non è decomposizione ma la chat di gruppo del primo conto con
un altro nome, e il conto è quello.

Secondo: serve un giudizio indipendente da chi ha prodotto. Qui il valore
aggiunto sta tutto nell'indipendenza: chi ha scritto il codice
ha già deciso, mentre lo scriveva, che è giusto. Separare chi fa da chi
controlla (nel loop engineering, *maker* e *checker*) serve esattamente a
questo, e il guadagno si può mettere in numeri.

`````{tab} Elementare

Rileggersi il proprio tema non serve quanto farlo rileggere a un compagno.
Rileggendoti, gli errori che non hai visto mentre scrivevi continui a non
vederli: sono proprio quelli su cui, scrivendo, avevi già deciso che andavano
bene. Un compagno che non sa che cosa avevi in testa li vede.

Mettiamoci dei numeri, scelti per fissare le idee e non misurati sul campo.
Diciamo che chi lavora sbaglia cinque volte su cento. Se poi si rilegge da solo
se ne accorge una volta su dieci: nove errori su dieci gli sfuggono e arrivano
in fondo, quindi di cinque su cento ne restano quattro e mezzo. Se invece a
rileggere è un altro, che parte dal risultato e non conosce il ragionamento che
c'è dietro, a sfuggirgli sono tre errori su dieci invece di nove: in fondo ne
arriva uno e mezzo su cento, tre volte meno. Stesso modello, stessa bravura,
stesso numero di controlli: cambia solo chi controlla.

Qui sta la cosa da non sbagliare mai. Quel vantaggio non viene dall'avere due
teste, viene dal fatto che la seconda non sa come è stato fatto il lavoro. Se al
controllore si passa tutto il ragionamento del primo, la sua scrivania torna a
essere quella del primo, e il vantaggio sparisce insieme all'ignoranza. E un
limite resta anche col compagno più ignaro: se ha studiato sugli stessi appunti,
gli errori che vengono da lì non li vede nemmeno lui.

`````

`````{tab} Superiore

Detta $P(E)$ la probabilità che il produttore sbagli e $P(M \mid E)$ quella che
il controllo non se ne accorga *dato che* l'errore c'è, un errore arriva in
fondo con probabilità $P(E)\,P(M \mid E)$. Poniamo $P(E) = 0{,}05$ e, per
fissare le idee, un autocontrollo che si accorge del proprio errore una volta su
dieci ($P(M \mid E) \approx 0{,}9$): lascia passare il $4{,}5\%$. Un critico con
contesto pulito e criteri propri scende a $P(M \mid E) \approx 0{,}3$ e lascia
passare l’$1{,}5\%$, tre volte meglio a parità di modello. Le due cifre sono
plausibili, non misurate; quello che conta è il verso della disuguaglianza, che
non dipende da quanto valgono esattamente. Tutto il valore del secondo agente
sta in quel condizionamento, e sparisce se gli si passa la trascrizione del
primo: a quel punto i due contesti tornano a coincidere. Il condizionamento
toglie però la correlazione che viene dal contesto, non quella che viene dai
pesi. Se produttore e controllore sono lo stesso modello, gli errori che
nascono da una lacuna del modello restano comuni ai due, e $P(M \mid E)$ non
scende sotto la quota di quegli errori, come avverte anche la
{doc}`sezione sul loop engineering </IngegneriaLLM/loop-engineering>`; per
abbassarla serve un controllore di un altro modello, o un controllo che non sia
un modello (un test, un tipo, un vincolo).

`````

Terzo: i contesti sono in conflitto. Qui il multi-agente è un modo di gestire
il contesto, prima che di aggiungere capacità.

`````{tab} Elementare

Prova a fare due lavori diversi sulla stessa scrivania, con le carte di
entrambi mescolate. La tua intelligenza è la stessa, ma ogni volta che
cerchi un foglio ne trovi tre dell'altra pratica, e ogni tanto scrivi in un
documento una cosa che riguardava l'altro. Due scrivanie, una per pratica,
risolvono il problema senza che nessuno diventi più bravo.

Con gli agenti succede lo stesso, e per una ragione precisa: tutto quello che
finisce nella finestra di contesto (la scrivania del modello, cioè il testo
che si rilegge davanti prima di rispondere) continua a influenzare le risposte
successive, anche quello che si è rivelato sbagliato. Un'ipotesi tentata e
abbandonata al terzo turno resta scritta lì, e al ventesimo turno tira ancora
la risposta dalla sua parte. Dare a ogni agente una finestra pulita sul suo
pezzo serve ad avere scrivanie separate, più che cervelli in più.

Che costano anche meno. Nella chat di gruppo ognuno rilegge le carte di tutti, e
il tempo di lettura si accumula; qui nessuno apre la pratica dell'altro, e
quattro scrivanie costano quattro volte una scrivania sola, non dieci. Ogni pila
resta più bassa, e quello che serve non finisce sepolto in mezzo agli altri
fogli.

Si paga da un'altra parte. Chi sta a una scrivania non sa che cosa è successo
all'altra, e le poche cose che devono passare di là vanno scelte a mano, una per
una. Quello che nessuno porta di là, l'altro non lo saprà mai.

`````

`````{tab} Superiore

L'argomento si legge nel conto dei token, girato al contrario.
Con $N$ agenti a contesti separati, ciascuno con la propria finestra e
nessuna trascrizione condivisa (solo una sintesi che attraversa il confine
alla fine), il costo totale è $N$ volte quello di un agente singolo:

$$
\text{token}_{\text{separati}}(N, R)
\;=\; N\left[R\,c_0 + \bar{m}\,\frac{R(R+1)}{2}\right]
\;=\; O(N R^{2}),
$$

lineare in $N$ anziché quadratico. Con i numeri di prima ($N = 4$, $R = 8$,
$c_0 = 2000$, $\bar{m} = 500$) si passa da $328.000$ token a
$4 \times 34.000 = 136.000$: quasi due volte e mezzo meno, e il rapporto
rispetto al singolo agente torna a essere esattamente $4\times$, cioè quello
che l'intuizione si aspettava fin dall'inizio. In più la finestra massima
scende da $18.000$ a $6.000$ token, tre volte più stretta, il che sposta ogni
agente fuori dal regime in cui il *lost in the middle* morde.

Il punto architetturale generale è che la topologia decide l'esponente: la
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
raccogliere aneddoti hanno letto dall'inizio alla fine le trascrizioni di
sistemi multi-agente al lavoro, cioè tutto quello che gli agenti si erano detti
e avevano fatto.

Il lavoro è andato in due tempi. Prima sei esperti hanno letto centocinquanta
trascrizioni prodotte con cinque framework diffusi, cioè cinque delle librerie
con cui queste squadre si costruiscono, e ne hanno ricavato un catalogo dei modi
di fallire. Che il catalogo si applicasse allo stesso modo da chiunque lo hanno
misurato: su quindici trascrizioni tre di loro, ciascuno per conto suo, finivano
per dare quasi sempre lo stesso giudizio (l'indice di accordo che si usa in
questi casi, il kappa di Cohen, vale 0,88, dove 1 è l'accordo perfetto e 0
quello che darebbe il caso). Poi il catalogo è stato affidato a un modello di
linguaggio, controllato prima sui casi già giudicati dagli esperti (accordo
0,77), che ha classificato 1.642 trascrizioni di sette framework. Il catalogo si
chiama MAST, una *tassonomia*, cioè una classificazione ordinata, dei fallimenti
dei sistemi multi-agente: quattordici modi ricorrenti raggruppati in tre
famiglie, e sono le famiglie la parte da ricordare, perché dicono *dove*
guardare.

La prima è quella delle **specifiche e del progetto del sistema**. L'agente non
rispetta un vincolo che il compito gli aveva posto, o esce dal ruolo assegnato,
o ripete un passo già fatto, o non riconosce la condizione in cui deve fermarsi.
Il tratto comune è che il difetto non dipende soltanto dal modello: dipende
anche da come la squadra è stata montata, cioè dai ruoli, dal flusso di lavoro e
da ciò che è stato scritto (o non scritto) all'inizio. È la più numerosa delle
tre famiglie, con il 44,2% dei fallimenti classificati, e la meno spettacolare.

La seconda è il **disallineamento fra agenti**. Ognuno ha in testa una sua
versione di come stanno le cose, le versioni si allontanano, e nessuno se ne
accorge perché la conversazione resta perfettamente fluente. Qui stanno l'agente
che ignora quello che gli è stato appena detto, quello che procede su una
richiesta ambigua senza chiedere chiarimenti, quello che trattiene
un'informazione utile agli altri, e la conversazione che scivola su un compito
diverso senza che nessuno lo dichiari. È la famiglia più insidiosa perché non
produce sintomi leggibili: due agenti in disaccordo su cosa stanno facendo si
scrivono messaggi cortesi e ben formati fino in fondo. Vale il 32,3% dei
fallimenti.

La terza è la **verifica e la terminazione**, cioè il momento in cui il sistema
decide di aver finito (23,5%). Il sistema si ferma troppo presto, oppure la
verifica manca del tutto, oppure c'è ma è inadeguata: il critico approva
perché non ha modo di provare davvero, e il suo verdetto non aggiunge
informazione, aggiunge una firma. Torna qui, misurata sui sistemi veri, la
matematica di poche righe fa: un controllo che intercetta quasi nessuno degli
errori lascia la catena esattamente dov'era, e resta un controllo di facciata.

Le tre famiglie chiedono interventi diversi. Per la prima si scrivono specifiche
più strette: i ruoli, le condizioni di arresto, chi ha l'ultima parola. Per la
seconda si rende esplicito lo stato condiviso invece di lasciarlo implicito
nella conversazione, cioè si scrive in un posto che tutti leggono che cosa è
stato deciso e che cosa resta da fare (è il mestiere dei protocolli e dei
meccanismi di consenso, a cui è dedicata la sezione su {doc}`protocolli e
consenso </SistemiMultiAgente/protocolli-e-consenso>`). Ma per questa famiglia
gli autori avvertono che protocolli e formati comuni non bastano: l'errore
compare anche fra agenti che nello stesso framework si scrivono in linguaggio
naturale, e la cura passa in parte per il modello, per la sua capacità di
rappresentarsi che cosa sa e che cosa serve all'altro. Per la terza si dà al
verificatore un criterio che possa davvero applicare. Nelle prove di intervento
degli autori su ChatDev, una squadra che simula una piccola azienda di
programmazione, con lo stesso modello e su trentadue compiti, ruoli scritti
meglio (e l'ultima parola a chi spetta) portano i compiti riusciti dal 25% al
34,4%, e una verifica dell'obiettivo prima di chiudere li porta al 40,6%: aiuta
molto, e non basta.

## La regola prudente

Il capitolo sugli Agenti chiudeva con una formula da rendere verificabile: si
aggiunge un ruolo solo quando risolve un problema reale, cioè un problema che
un agente da solo non risolveva. La versione operativa ha quattro clausole.

Il problema deve essere documentato, non intuito. Prima di aggiungere il
critico bisogna poter esibire i casi in cui il solista sbagliava e il critico
avrebbe intercettato. Se quei casi non si trovano fra le trascrizioni di quello
che il sistema ha già fatto, il ruolo sta risolvendo un problema immaginario.

Il confronto è contro un singolo agente ben progettato. È la clausola che
salta più spesso, e senza di essa qualunque architettura vince. Un solista con
un buon foglio di istruzioni, gli strumenti giusti e un cancello di verifica è
un avversario ben diverso da un solista improvvisato, ed è contro di lui che il
conto va fatto. Il lavoro sui fallimenti reali apre proprio dichiarando che
sulle prove di confronto più diffuse il guadagno delle squadre resta spesso
minimo rispetto ai sistemi a un agente solo, e minimo perfino rispetto a un
paragone elementare: chiedere la stessa cosa più volte allo stesso modello e
tenere la risposta migliore secondo un verificatore {cite}`cemri2025why`. Ed è
una letteratura giovane, ancora lontana da un verdetto netto {cite}`xi2023rise`.

A parità di budget. Se la squadra consuma 328.000 token, il termine di
paragone onesto non è il solista da 34.000: è il solista a cui si danno gli
stessi 328.000, spesi facendogli rifare la stessa domanda molte volte e tenendo
la risposta che esce più spesso, o lasciandolo ragionare più a lungo, o dandogli
più tentativi contro il cancello di verifica. Nei confronti fatti così, i
dibattiti fra più agenti non battono in modo affidabile strategie più semplici
dello stesso modello, come la risposta più frequente su più ragionamenti
campionati {cite}`smit2024should`: una parte dei guadagni attribuiti al
multi-agente è calcolo in più. Alcuni protocolli, regolati con cura, superano
quella base, quindi il confronto va rifatto sul proprio compito.

Il ruolo si tiene solo se una misura lo conferma. La prova è semplice: si
toglie il ruolo, si rimisura, e se il numero non si muove il ruolo esce. Le
misure sono quelle già viste per gli agenti, e vanno lette tutte insieme e mai
una sola: quante volte il compito riesce, quanto è pulita la strada che il
sistema ha fatto per arrivarci, quanto costa e quanto fa aspettare. Perché
un'architettura che alza il successo di due punti e il costo di dieci volte non
ha vinto, ha speso.

Sotto tutte e quattro c'è la lezione di Brooks, mezzo secolo dopo: il
coordinamento è una voce di costo, e con una trascrizione condivisa cresce più
in fretta (con il quadrato dei turni) del beneficio che dovrebbe pagare, che la
legge di Amdahl tiene sotto un tetto. Un sistema multi-agente ben progettato ha
il minimo numero di agenti che risolve il problema, ciascuno con una finestra
pulita e un compito che il solista non chiudeva.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Con una trascrizione condivisa (la chat di gruppo in cui, prima di
  scrivere, si rilegge tutto dall'inizio) a ogni raddoppio degli interventi le
  righe da leggere si quadruplicano, perché ogni messaggio nuovo dovrà essere
  riletto da tutti quelli che parleranno dopo. Sul conto vero, che comprende
  anche le istruzioni di partenza, la crescita è un po’ più mite ma resta
  spaventosa: quattro agenti per otto giri leggono 328.000 token contro i 34.000
  di un solista, cioè quasi dieci volte, non quattro.
- La parte di lavoro che non si può dividere mette un tetto: è la legge di
  Amdahl. Se su dieci ore di rapporto tre servono a decidere la scaletta e a
  rileggere alla fine, nessun numero di collaboratori chiude quel rapporto in
  meno di tre ore, e più veloci di tre volte e un terzo non si va. Anche quel
  poco, poi, si assottiglia in fretta: da quattro persone a otto si guadagna meno
  di un'ora su dieci. E a quel tetto non ci si arriva nemmeno, perché ogni
  collaboratore in più fa perdere tempo anche agli altri: contando anche quello,
  gli otto guadagnano pochi minuti sui quattro, e oltre la sesta persona assumere
  allunga il lavoro invece di accorciarlo.
- Gli errori si moltiplicano, non si mediano: è il telefono senza fili. Con
  partecipanti che riferiscono bene novantacinque volte su cento, dieci passaggi
  lasciano la frase intatta sei volte su dieci e venti passaggi poco più di tre.
  Basta però un controllo esterno a ogni passaggio (il cancello di verifica)
  che intercetti otto errori su dieci e faccia rifare il passaggio, e i venti
  passaggi tornano a riuscire otto volte su dieci: la verifica esterna è
  ciò che spezza la catena.
- Si guadagna davvero in pochi casi, e i conti ne sanno valutare tre: il
  compito si spezza in parti che non hanno bisogno di parlarsi mentre lavorano;
  serve un giudizio indipendente da chi ha prodotto (il valore sta nel non aver
  già deciso, non nella potenza aggiunta); due lavori si disturbano a vicenda e
  conviene dare a ciascuno la sua scrivania, cioè un contesto pulito. In
  quest'ultimo caso, per giunta, il conto smette di esplodere: nessuno legge la
  roba degli altri, e quattro agenti costano quattro volte un solista invece di
  dieci. Un quarto i conti non lo misurano: separare i permessi, perché chi
  scrive il codice non sia chi lo esegue.
- I fallimenti reali {cite}`cemri2025why` si raggruppano in tre famiglie:
  specifiche e progetto (un vincolo del compito o del ruolo non rispettato, un
  passo rifatto per niente: è la più numerosa), disallineamento fra agenti
  (ognuno ha una versione diversa di che cosa sta succedendo, sotto una
  conversazione perfettamente fluente: qui formati e protocolli aiutano ma non
  bastano), verifica e terminazione (si approva senza poter davvero
  controllare, o ci si ferma troppo presto).
- La regola prudente: il problema va documentato e non intuito, il confronto
  si fa contro un singolo agente ben progettato e a parità di token
  spesi, e il ruolo si tiene solo se togliendolo il risultato peggiora davvero.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Con una trascrizione condivisa il costo va come il quadrato dei turni,
  $\sum_{t=1}^{T}(c_0 + t\bar{m}) = O(\bar{m}T^2)$, e siccome $T = NR$ è
  quadratico anche negli agenti. Quattro agenti per otto giri leggono
  $328.000$ token contro i $34.000$ di un solista: quasi dieci volte, non
  quattro.
- La legge di Amdahl mette un tetto: con una frazione seriale $s$
  l'accelerazione è al più $1/(s + (1-s)/N)$, cioè $1/s$ per $N \to \infty$.
  Con $s = 0{,}3$ nessun numero di agenti supera $3{,}3\times$; e con una
  penale di coordinamento la curva torna giù dopo $N^{*}=\sqrt{(1-s)/\kappa}$.
- Gli errori si compongono: $n$ passi corretti con probabilità $p$ danno
  $p^n$. Con $p = 0{,}95$, dieci passi danno $0{,}60$ e venti $0{,}36$. Un
  validation gate che intercetta l’$80\%$ degli errori, con il passo
  intercettato rifatto con successo, porta $p' = p + (1-p)\,r\,q$ a $0{,}99$ e
  i venti passi a $0{,}82$ (a $0{,}59$ se il rifacimento riesce con
  $q = 0{,}6$): il gate è ciò che spezza la catena moltiplicativa.
- Si guadagna davvero in pochi casi, e tre si sanno valutare: il compito si
  decompone in parti quasi indipendenti (e se la taglia cresce con $N$ vale
  Gustafson, non Amdahl); serve un giudizio indipendente (il valore è nella
  decorrelazione di contesto, che non tocca gli errori dei pesi); i contesti
  sono in conflitto e conviene separarli, e allora il multi-agente è gestione
  del contesto (costo $O(NR^2)$ invece di $O(N^2R^2)$: la topologia decide
  l'esponente). Un quarto, la separazione dei permessi, è di sicurezza e non
  compare nei conti.
- I fallimenti reali {cite}`cemri2025why` si raggruppano in tre famiglie, su
  1.642 trascrizioni di sette framework: progetto del sistema (vincoli di
  compito e di ruolo disattesi, passi ripetuti, condizioni di arresto non
  riconosciute: 44,2%), disallineamento fra agenti (stati del mondo divergenti
  sotto una conversazione fluente: 32,3%, e qui protocolli e formati non
  bastano), verifica e terminazione (si approva senza poter provare, o ci si
  ferma troppo presto: 23,5%).
- La regola prudente: problema documentato, confronto contro un singolo
  agente ben progettato e a parità di budget di token (dove il dibattito non
  batte in modo affidabile l'autoconsistenza {cite}`smit2024should`), e il
  ruolo si tiene solo se l'ablazione lo conferma.
```

`````

Il costo, insomma, dipende da chi legge che cosa, e chi legge che cosa lo decide
la forma della comunicazione fra gli agenti: chi scrive a chi, e chi non scrive
a nessuno. Quelle forme sono poche e ricorrenti, e la {doc}`sezione su chi
parla con chi <topologie>` le mette a confronto.
