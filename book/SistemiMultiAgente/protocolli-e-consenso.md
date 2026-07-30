# Mettersi d'accordo: dire, votare, diffidare

Nel 1955 il filosofo del linguaggio John Langshaw Austin tenne a Harvard un
ciclo di lezioni che sarebbe uscito in volume solo dopo la sua morte, con un
titolo che è già una tesi: *How to Do Things with Words*, come fare cose con le
parole. L'osservazione di partenza sembra ovvia appena qualcuno la pronuncia.
Certe frasi descrivono il mondo, e di esse ha senso chiedersi se siano vere o
false: «piove». Altre non descrivono niente, lo cambiano. «Prometto di venire
domani» non è né vero né falso: è un impegno che un istante prima non
esisteva. Il sindaco che dice «vi dichiaro marito e moglie» non riferisce di un
matrimonio, lo celebra. Austin chiamò queste frasi **enunciati performativi**, e
da lì è nata la teoria degli atti linguistici, il capitolo della pragmatica che
studia ciò che le parole *fanno* oltre a ciò che dicono.

Sembra filosofia, ed è ingegneria. Quando due agenti si scambiano un
messaggio, la domanda «che cosa c'è scritto» è la meno interessante; quella
che conta è «che cosa fa questo messaggio»: apre un impegno, ne chiude uno,
chiede, propone, rifiuta. Subito dopo arriva la seconda domanda, quella che
con un agente solo non si pone: quando le risposte non coincidono, **chi
decide**? Questa sezione percorre le risposte che il campo ha dato, dal
messaggio tipizzato al voto di maggioranza, e si ferma a lungo sulla crepa che
sta sotto il voto, perché è il punto più frainteso dell'intero capitolo.

## Un messaggio è una mossa

Cominciamo dalla forma. In un sistema multi-agente il testo che passa da un
partecipante all'altro non è prosa: è una mossa dentro una partita, e come ogni
mossa ha un *tipo*.

`````{tab} Elementare

Immagina tre biglietti attaccati al frigorifero. Sul primo c'è scritto «il
latte è finito». Sul secondo «compra il latte». Sul terzo «il latte lo prendo
io tornando». Parlano tutti e tre della stessa cosa, con quasi le stesse
parole, ma fanno tre mestieri diversi: il primo dà una notizia, il secondo
scarica un compito su chi legge, il terzo se lo prende chi ha scritto.

La differenza si vede domani mattina, quando il latte manca ancora. Col primo
biglietto non ha sbagliato nessuno: era solo un'informazione. Col secondo hai
sbagliato tu. Col terzo ha sbagliato chi l'ha scritto. Stesso argomento, tre
responsabilità diverse, e a stabilirlo non è l'argomento ma il *tipo* di
biglietto. Tenerlo scritto sul biglietto, invece di lasciarlo indovinare a chi
legge, è l'unica cosa che permette a fine giornata di dire se qualcuno ha
mancato a qualcosa.

`````

`````{tab} Superiore

Austin distingue tre livelli in ogni enunciato: l'atto **locutorio** (dire
qualcosa di sensato), quello **illocutorio** (ciò che si compie *nel* dirlo:
promettere, ordinare, asserire) e quello **perlocutorio** (l'effetto ottenuto
*col* dirlo: convincere, spaventare). Il livello che ci serve è il secondo.
John Searle lo formalizza scomponendo ogni enunciato in due coordinate
indipendenti, $F(P)$: la **forza illocutoria** $F$ e il **contenuto
proposizionale** $P$. «Chiudi la porta» e «prometto di chiudere la porta»
condividono $P$ e differiscono per $F$, e non c'è modo di ricavare $F$ da $P$:
sono assi ortogonali.

Searle raggruppa poi le forze in cinque classi: **assertivi** (impegnano chi
parla sulla verità di $P$), **direttivi** (tentano di far fare qualcosa a chi
ascolta), **commissivi** (impegnano chi parla a un'azione futura),
**espressivi** e **dichiarazioni** (che modificano il mondo per il solo fatto
di essere pronunciate da chi ne ha l'autorità). A ciascuna classe
corrispondono condizioni che Austin chiamava **di felicità** e che Searle
sistema in condizioni preparatorie, di sincerità ed essenziali: un ordine dato
a chi non è tenuto a obbedire non è falso, è nullo. Ed è proprio la distinzione
fra «falso» e «nullo» che serve a un sistema distribuito: un contenuto
sbagliato è un problema di merito, un atto mal formato è un problema di
protocollo, e si diagnosticano in modi diversi.

`````

Negli anni Novanta questa analisi diventa un tipo di dato. **KQML**
(*Knowledge Query and Manipulation Language*), sviluppato da Tim Finin e
colleghi nell'ambito del Knowledge Sharing Effort finanziato dalla DARPA, e
poi **FIPA-ACL**, lo standard della Foundation for Intelligent Physical Agents
nata nel 1996, stabiliscono che ogni messaggio fra agenti porti in chiaro la
propria **performativa**. Non è una convenzione stilistica: è un campo
obbligatorio, scelto in un elenco finito (ventidue voci nella libreria degli
atti comunicativi di FIPA: `inform`, `request`, `agree`, `refuse`, `propose`,
`accept-proposal`, `failure`, `not-understood` e le altre), accanto a
mittente, destinatario, contenuto, identificativo della conversazione e
riferimento al messaggio a cui si risponde.

Perché conta oggi, che gli agenti sono modelli di linguaggio e la prosa libera
gli riesce benissimo? Perché la prosa libera non si verifica. Se un agente
scrive «ci penso io, più tardi», nessun programma può stabilire se ha accettato
un incarico o se sta rimandando, e a fine esecuzione nessuno può dire a macchina
se quella richiesta ha avuto risposta, se quell'impegno è stato onorato, se
quella proposta è stata accettata o ignorata. Tipizzare i messaggi trasforma la
conversazione in una **macchina a stati ispezionabile**, che si registra, si
riesegue e si controlla. Bastano poche righe.

```python
from dataclasses import dataclass


@dataclass
class Messaggio:
    """La performativa dice che cosa il messaggio *fa*, non di che cosa parla."""
    performativa: str   # richiedi, accetta, rifiuta, informa, fallisci
    mittente: str
    destinatario: str
    filo: str           # a quale conversazione appartiene
    contenuto: str


# La macchina a stati del protocollo: da ogni stato, quali performative sono
# lecite e in quale stato portano.
TRANSIZIONI = {
    "aperto":    {"richiedi": "in attesa"},
    "in attesa": {"accetta": "impegnato", "rifiuta": "chiuso"},
    "impegnato": {"informa": "chiuso", "fallisci": "chiuso"},
    "chiuso":    {},
}

# Gli stati in cui la conversazione non e' finita: sono i conti aperti.
IN_SOSPESO = {"in attesa": "<- richiesta senza risposta",
              "impegnato": "<- impegno non onorato"}


def ripercorri(traccia):
    """Rilegge la traccia e restituisce lo stato finale di ogni filo.
    Solleva un errore alla prima mossa che il protocollo non prevede."""
    stati = {}
    for m in traccia:
        stato = stati.get(m.filo, "aperto")
        lecite = TRANSIZIONI[stato]
        if m.performativa not in lecite:
            raise ValueError(
                f"filo {m.filo}: '{m.performativa}' non e' lecita nello stato "
                f"'{stato}' (attese: {sorted(lecite) or 'nessuna'})")
        stati[m.filo] = lecite[m.performativa]
    return stati


traccia = [
    Messaggio("richiedi", "pianificatore", "ricercatore", "f1", "trova i dati"),
    Messaggio("accetta",  "ricercatore", "pianificatore", "f1", "procedo"),
    Messaggio("richiedi", "pianificatore", "analista", "f2", "stima il trend"),
    Messaggio("informa",  "ricercatore", "pianificatore", "f1", "ecco la tabella"),
    Messaggio("accetta",  "analista", "pianificatore", "f2", "procedo"),
    Messaggio("richiedi", "pianificatore", "revisore", "f3", "controlla la stima"),
    Messaggio("rifiuta",  "revisore", "pianificatore", "f3", "mi manca la tabella"),
]

for filo, stato in sorted(ripercorri(traccia).items()):
    print(f"{filo}: {stato:10s} {IN_SOSPESO.get(stato, '')}".rstrip())

# Una mossa fuori protocollo viene intercettata subito.
fuori = Messaggio("informa", "revisore", "pianificatore", "f3", "ecco il controllo")
try:
    ripercorri(traccia + [fuori])
except ValueError as errore:
    print("violazione:", errore)
```

```text
f1: chiuso
f2: impegnato  <- impegno non onorato
f3: chiuso
violazione: filo f3: 'informa' non e' lecita nello stato 'chiuso' (attese: nessuna)
```

Una trentina di righe di macchina a stati, senza una sola chiamata a un
modello, eppure il sistema sa dire due cose che nessuna rilettura della
trascrizione avrebbe dato gratis: che
l'analista ha promesso una stima e non l'ha mai consegnata, e che il revisore
ha provato a rispondere su un filo che aveva già chiuso rifiutando. La prima è
un **impegno pendente**, la seconda una **violazione di protocollo**, e sono
guasti diversi: al primo si rimedia con un sollecito o un timeout, al secondo
scartando il messaggio. Il disallineamento fra agenti, che la sezione sul costo
del coordinamento elencava fra le tre famiglie di fallimento
{cite}`cemri2025why`, si manifesta quasi sempre così, sotto una conversazione
perfettamente cortese, e il tipo di messaggio è il primo strumento che lo rende
visibile.

È lo stesso movimento dell'**output strutturato** del capitolo sull'ingegneria
degli LLM: si restringe la forma di ciò che il modello può produrre in cambio
della possibilità di controllarlo a macchina. Cambia la scala. Il *validation
gate* del loop engineering è un predicato su una singola uscita; un protocollo
è un predicato sull'**intera conversazione**: non «questa risposta è ben
formata», ma «questo scambio, dal primo messaggio all'ultimo, è una partita
legale».

Il capostipite di questi protocolli ha più di quarant'anni, ed è lo stesso
**Contract Net** di Reid G. Smith {cite}`smith1980contract` che la sezione
sulle topologie ha incontrato come mercato. Là interessava chi prende il
lavoro; qui interessa la forma dello scambio. Bando, offerta e assegnazione
sono tre atti tipizzati in una sequenza fissa, e alla fine della sequenza
esiste un oggetto che prima non esisteva: un contratto, con un responsabile e
una scadenza. È la stessa aritmetica dei biglietti sul frigorifero, portata su
scala di sistema: dopo l'aggiudicazione la domanda «questo compito ha un
titolare?» ha una risposta che il programma sa dare da sé, senza rileggere
niente. Per questo la struttura si ritrova, di rado citata, in ogni
orchestratore moderno che chiede a più agenti se sono in grado di svolgere un
compito prima di affidarlo.

## Aggregare i giudizi: il conto di Condorcet

Dai messaggi passiamo alle decisioni. Se più agenti hanno risposto e le
risposte differiscono, la strada più ovvia è contare: vince la maggioranza. Ed
è una strada con alle spalle il teorema più antico della materia. Nel 1785
Nicolas de Condorcet, matematico e politico, pubblica un saggio
sull'applicazione del calcolo delle probabilità alle decisioni prese a
maggioranza di voti. La domanda era concreta e rivoluzionaria: una giuria
numerosa giudica meglio di un singolo giudice? La risposta è sì, a due
condizioni, e sono le condizioni a interessarci.

`````{tab} Elementare

Tre persone rispondono a una domanda difficile e ciascuna, da sola, ci prende
sette volte su dieci. Elenchiamo tutti i casi possibili, con la loro
probabilità.

Che ci prendano tutte e tre capita $0{,}7 \times 0{,}7 \times 0{,}7 = 0{,}343$
delle volte. Che ne azzecchino esattamente due (e la terza sbagli) capita in
tre modi diversi, a seconda di chi sbaglia, e ciascun modo vale
$0{,}7 \times 0{,}7 \times 0{,}3 = 0{,}147$: in tutto $0{,}441$. Che ne
azzecchi una sola: tre modi da $0{,}7 \times 0{,}3 \times 0{,}3 = 0{,}063$,
cioè $0{,}189$. Che sbaglino tutte: $0{,}3^3 = 0{,}027$. I quattro numeri
sommati fanno $1$, come devono.

La maggioranza ha ragione nei primi due casi, quando ci prendono in tre o in
due: $0{,}343 + 0{,}441 = 0{,}784$. Il gruppo azzecca quasi otto volte su
dieci mentre ciascuno, da solo, ne azzecca sette. Il guadagno viene da un
fatto elementare: perché il gruppo sbagli servono **almeno due errori
insieme**, e due errori insieme sono più rari di uno. Con cinque persone
servirebbero tre errori insieme e la maggioranza arriva a $0{,}837$; con nove
ne servirebbero cinque, e si arriva a $0{,}901$.

`````

`````{tab} Superiore

È il **teorema della giuria di Condorcet**. Con $n$ votanti che decidono in
modo **indipendente**, ciascuno corretto con probabilità $p$, e regola di
maggioranza semplice, la probabilità che il verdetto collettivo sia corretto è

$$
P_n = \sum_{k=\lfloor n/2 \rfloor + 1}^{n} \binom{n}{k}\, p^{k}\,(1-p)^{\,n-k},
$$

dove $\binom{n}{k}$ conta i modi in cui $k$ votanti su $n$ possono essere
quelli corretti, $p^k(1-p)^{n-k}$ è la probabilità di ciascuno di quei modi e
l'estremo inferiore della somma è la più piccola maggioranza stretta (per $n$
dispari, $(n+1)/2$; si prende $n$ dispari proprio per non dover arbitrare i
pareggi). Il teorema ha due parti: per $p > 1/2$ la successione $P_n$, letta
sui valori dispari di $n$, è crescente e tende a $1$; per $p < 1/2$ è
decrescente e tende a $0$. Il voto non aggiunge competenza, **amplifica la
tendenza di fondo**, qualunque essa sia: sopra la soglia converge alla verità,
sotto la soglia converge all'errore, e la soglia è esattamente il tirare a
caso.

Con $p = 0{,}7$: $P_3 = 0{,}784$, $P_5 = 0{,}837$, $P_9 = 0{,}901$,
$P_{21} = 0{,}974$. La convergenza è reale ma lenta, con rendimenti
decrescenti marcati: i primi due votanti aggiunti comprano otto punti
($0{,}700 \to 0{,}784$), i dodici che portano da nove a ventuno ne comprano
sette ($0{,}901 \to 0{,}974$). Vale la pena confrontare questa curva con il
costo, che l'apertura del capitolo ha mostrato crescere come il quadrato
quando tutti leggono tutto: il voto è il caso migliore per il multi-agente
proprio perché i votanti non si parlano, e quindi il costo resta lineare in
$n$.

`````

## L'ipotesi che non regge

Qui arriva il punto della sezione, e conviene dirlo senza attenuanti. Il
teorema di Condorcet ha un'ipotesi, l'**indipendenza**, e nei sistemi
multi-agente costruiti oggi quell'ipotesi è quasi sempre falsa.

Dieci istanze dello stesso modello, con lo stesso prompt di sistema, davanti
alla stessa domanda, non sono dieci votanti: sono **un votante interrogato
dieci volte**. Non è un sospetto, è una conseguenza di come sono fatte. Un
modello addestrato è una funzione con dei parametri fissi: se si spegne il caso
nella generazione (la temperatura a zero) dieci istanze restituiscono la stessa
identica uscita, e la correlazione fra i voti è $1$ per costruzione. Alzando la
temperatura si campiona intorno a quella funzione, non se ne ottiene una
diversa: cambia il percorso di generazione, restano identici i pesi, i dati su
cui il modello si è addestrato e le lacune che quei dati hanno lasciato. Un
errore **sistematico** vive esattamente lì (una formula memorizzata male,
un'ambiguità letta sempre nello stesso verso, un fatto che nei testi di
addestramento compare solo nella versione sbagliata) e non è il tipo di errore
che il campionamento disperde: non è che qualcuno degli agenti sbagli,
sbagliano tutti, e sbagliano **allo stesso modo**. La maggioranza, in quel
caso, non corregge niente: certifica.

È anche una previsione che si può mettere alla prova in mezz'ora, e conviene
farlo prima di fidarsi di un voto: si prendono le domande su cui il sistema ha
sbagliato e si conta quanto spesso, su quelle, gli agenti erano d'accordo fra
loro. Se l'accordo sugli errori è alto, i voti non erano indipendenti e la
formula di Condorcet non si applica.

Mettiamo dei numeri, con il modello più semplice che catturi il fenomeno. Una
frazione $\lambda$ delle domande sono *trappole*: contengono la caratteristica
che manda fuori strada quel modello, e su di esse sbagliano tutti gli agenti
insieme. Sulle altre gli errori sono indipendenti come vuole Condorcet.

`````{tab} Elementare

Diciamo che una domanda su cinque è una trappola, e che sulle altre quattro
ogni agente ci prende sette volte su otto. Sono numeri scelti apposta perché il
singolo agente, sul totale delle domande, resti corretto le solite sette volte
su dieci: uguale a prima, indistinguibile da fuori se guardi un agente alla
volta.

Cambia tutto quando voti. Con tre agenti passi da $0{,}784$ a $0{,}766$: poco.
Con nove agenti, dove Condorcet prometteva $0{,}901$, ti fermi a $0{,}798$. E
la cosa da guardare non è quanto hai perso, è che **oltre non si va**: con
ventuno agenti fai $0{,}800$, con novantanove ancora $0{,}800$. Il tetto lo
fissa la quota di domande-trappola: restano le altre quattro su cinque, cioè
l'80%, e nessun numero di partecipanti va oltre, perché su quel quinto di
domande stanno sbagliando tutti insieme.

C'è di peggio, ed è la parte che dovrebbe far paura. Sulle trappole i nove
agenti non sbagliano un po' ciascuno per conto suo: rispondono la stessa cosa
sbagliata, **all'unanimità**. Se usi l'accordo come misura di fiducia (nove su
nove, andiamo tranquilli) stai leggendo il segnale più forte proprio nel
momento in cui è più falso. Fai il conto: quando tutti e nove concordano,
quasi una volta su due (il 45%) concordano su una risposta sbagliata.

`````

`````{tab} Superiore

Sia $\lambda$ la probabilità che una domanda appartenga alla regione di errore
sistematico, dove il voto degenera perché tutti gli agenti producono la stessa
risposta errata, e $p_0$ la probabilità di correttezza individuale sulle
domande restanti, dove gli errori sono indipendenti. La probabilità marginale
di correttezza del singolo agente è $p = (1-\lambda)\,p_0$, e quella del voto
di maggioranza vale

$$
P_n = (1-\lambda) \sum_{k=\lfloor n/2 \rfloor + 1}^{n}
      \binom{n}{k}\, p_0^{k}\,(1-p_0)^{\,n-k},
\qquad
\lim_{n \to \infty} P_n = 1 - \lambda,
$$

dove il fattore $1-\lambda$ non dipende da $n$ e mette un **tetto** che nessun
numero di votanti supera. Con $\lambda = 0{,}2$ e $p_0 = 0{,}875$ si ha
$p = 0{,}7$, identica al caso indipendente, ma il tetto è $0{,}8$ contro l'$1$
di Condorcet.

La versione generale del fenomeno si legge sulla varianza. Per $n$ votanti
scambiabili, ciascuno con varianza $\sigma^2$ e correlazione a coppie $\rho$,
la varianza della media vale $\sigma^2\,[1 + (n-1)\rho]/n$, che per $n$ grande
tende a $\sigma^2\rho$ invece che a zero. Il **numero efficace di votanti** è
quindi

$$
n_{\text{eff}} = \frac{n}{1 + (n-1)\rho}
\;\xrightarrow[n \to \infty]{}\; \frac{1}{\rho},
$$

dove $\rho$ è la correlazione fra i giudizi di due votanti qualsiasi. Con
$\rho = 0{,}5$, mille agenti valgono **due** votanti indipendenti; con
$\rho = 0{,}2$ ne valgono cinque. È la stessa aritmetica che governa gli
*ensemble* nel capitolo di machine learning, dove il guadagno del bagging viene
dalla decorrelazione e non dal numero di alberi. La conseguenza per chi
progetta è una sola: finché $\rho$ non si misura, il numero di agenti che si
pagano dice poco sul numero di giudizi indipendenti che si ottengono, e la
curva reale sta sotto quella di Condorcet.

Peggiora se si guarda l'unanimità. Nel modello sopra, con $n = 9$, la
probabilità che tutti concordino è $\lambda + (1-\lambda)p_0^9 = 0{,}441$: il
primo addendo sono le trappole, il secondo il caso in cui, fuori dalle
trappole, tutti e nove abbiano ragione (si trascura la coincidenza di nove
errori indipendenti, che su risposte non binarie ha probabilità irrisoria). La
probabilità che una risposta unanime sia **sbagliata** è dunque
$\lambda / 0{,}441 = 0{,}454$. L'accordo, usato come stima della confidenza, è
un cattivo stimatore in modo *asimmetrico*: è alto quando l'errore è
sistematico, cioè proprio nei casi in cui servirebbe un allarme.

`````

Il codice che produce questi numeri sta in venti righe e non ha bisogno di
nulla oltre la libreria standard.

```python
from math import comb


def maggioranza(n, p):
    """Probabilita' che piu' della meta' di n votanti indipendenti sia corretta."""
    return sum(comb(n, k) * p**k * (1 - p) ** (n - k)
               for k in range(n // 2 + 1, n + 1))


def maggioranza_correlata(n, p0, lam):
    """Con probabilita' lam la domanda e' una trappola e sbagliano tutti insieme;
    altrimenti ciascuno e' corretto in modo indipendente con probabilita' p0."""
    return (1 - lam) * maggioranza(n, p0)


lam, p0 = 0.2, 0.875   # cosi' il singolo agente resta corretto il 70% delle volte

print("  n   indipendenti   correlati")
for n in (1, 3, 5, 9, 21, 99):
    print(f"{n:3d}       {maggioranza(n, 0.7):.3f}         "
          f"{maggioranza_correlata(n, p0, lam):.3f}")

# Nove agenti che rispondono all'unisono: quanto vale quell'unanimita'?
concordi_giusti = (1 - lam) * p0**9
concordi_sbagliati = lam           # sulle trappole sbagliano tutti allo stesso modo
unanimi = concordi_giusti + concordi_sbagliati
print(f"\nP(unanimita' su 9) = {unanimi:.3f}, "
      f"di cui sbagliate {concordi_sbagliati / unanimi:.1%}")
```

```text
  n   indipendenti   correlati
  1       0.700         0.700
  3       0.784         0.766
  5       0.837         0.787
  9       0.901         0.798
 21       0.974         0.800
 99       1.000         0.800

P(unanimita' su 9) = 0.441, di cui sbagliate 45.4%
```

Riassunto in una riga: **fra agenti identici il voto non aumenta la
correttezza, aumenta la confidenza**. Ed è l'esito peggiore possibile, perché
un sistema che sbaglia e lo sa è recuperabile, mentre uno che sbaglia con nove
firme in calce non lo è.

L'indipendenza, però, non è tutto o niente: un po' se ne può comprare, e ogni
modo di comprarla ha il suo prezzo. Si può **campionare a temperatura non
nulla** invece di decodificare in modo greedy, così che i percorsi di
generazione divergano. Si può chiedere esplicitamente **percorsi di
ragionamento diversi** (risolvi per stima, poi per calcolo esatto, poi
verificando all'indietro). Si può **cambiare il modo in cui la domanda è
posta**, riformulandola o riordinando le opzioni, che è anche il modo di
scoprire quanto la risposta dipendeva dalla formulazione. E si può, quando è
possibile, **cambiare modello**: è l'intervento che decorrela di più, perché
tocca i pesi e non solo il campionamento, ed è anche il più caro da gestire.

È esattamente il meccanismo della **self-consistency**
{cite}`wang2023selfconsistency`: invece di prendere l'unica catena di
ragionamento prodotta dalla decodifica greedy, se ne campionano molte e si
marginalizza sul ragionamento, tenendo la risposta finale più frequente.
Funziona, e funziona per la ragione che questa sezione ha appena messo in
formula: il campionamento **decorrela parzialmente** i percorsi. Vale la pena
insistere sull'avverbio. I percorsi campionati condividono il modello, quindi
$\rho$ scende ma non va a zero, il tetto $1-\lambda$ resta dov'è, e il
guadagno reale è sempre inferiore a quello che il conto di Condorcet
promette. Chi progetta un sistema di voto dovrebbe misurare $P_n$ per $n$
crescente e guardare dove la curva **si appiattisce**: quel plateau è la stima
empirica del tetto, ed è il numero che dice quando smettere di pagare agenti in
più.

## Dibattere invece di votare

Se contare le teste non basta, si può cambiare gioco: invece di aggregare
risposte, farle scontrare. Nel **dibattito** due agenti sostengono posizioni
opposte sulla stessa domanda, si contestano a vicenda, e un terzo (un umano, o
un altro modello) decide chi ha argomentato meglio. La proposta, come impianto
di ricerca sull'allineamento, è di Geoffrey Irving, Paul Christiano e Dario
Amodei nel 2018 {cite}`irving2018ai`.

Il meccanismo interessante non è la gara: è l'**asimmetria fra produrre e
verificare**. Trovare la dimostrazione di un teorema è difficile;
controllarla, riga per riga, è molto più facile. Il dibattito sfrutta questo
scarto: converte un problema di generazione, che il giudice non saprebbe
affrontare, in un problema di verifica, che è alla sua portata.

`````{tab} Elementare

Pensa a un processo. Il giudice non ha svolto le indagini, non ha visitato la
scena, non ha interrogato i testimoni: da solo non arriverebbe mai alla
verità. Eppure decide, e il sistema funziona perché non deve *ricostruire*
niente: deve solo accorgersi di quando un ragionamento non sta in piedi. Ci
sono due parti che hanno interesse opposto, e ciascuna ha tutte le ragioni per
mettere in evidenza la falla dell'altra.

Ecco il punto che rende il meccanismo forte: se uno dei due mente, all'altro
conviene puntare il dito **esattamente** sul punto della bugia. E controllare
un solo punto è alla portata di chiunque, anche di chi non avrebbe saputo
ricostruire l'intera storia. Il giudice non deve essere più bravo dei
dibattenti: deve solo saper valutare l'ultimo passaggio contestato.

E qui sta anche il limite, che è bene guardare in faccia. Tutto regge finché
il giudice riconosce un argomento fallace da uno valido. Se si lascia
convincere dal più sicuro di sé, dal più fluente, da chi usa le parole
difficili, allora il dibattito non premia chi ha ragione: premia chi è più
persuasivo, che è una qualità diversa e a volte opposta.

`````

`````{tab} Superiore

Il dibattito si formalizza come un **gioco a somma zero** a due giocatori: dati
una domanda e un limite di lunghezza degli interventi, i due agenti si
alternano e alla fine un giudice, con risorse di calcolo limitate, dichiara chi
ha dato l'informazione più vera e utile. L'addestramento avviene per *self-play*
sullo stesso gioco, con la stessa struttura vista nel Reinforcement Learning e
ripresa nella sezione «Imparare insieme».

L'argomento teorico che gli autori portano è un'analogia con la teoria della
complessità, e chiarisce esattamente che cosa il dibattito compri
{cite}`irving2018ai`. Un giudice che valuta direttamente una risposta esibita
da un solo agente può decidere, con gioco ottimale, le domande in **NP**:
quelle per cui esiste un certificato breve, verificabile in tempo polinomiale.
Il gioco del dibattito, con due agenti in competizione e lo stesso giudice
polinomiale, arriva a **PSPACE**: una classe molto più ampia, perché
l'alternanza fra i due giocatori corrisponde all'alternanza dei quantificatori
in un gioco a informazione perfetta. Il guadagno non viene dal fatto che i
dibattenti siano più bravi, ma dal fatto che il giudice non deve esaminare
l'intero albero degli argomenti: gli basta seguire il ramo che i due, con
interessi opposti, hanno scelto di contestare.

Due avvertenze, e sono sostanziali. La prima: il risultato vale con gioco
ottimale e giudice affidabile, cioè capace di valutare correttamente l'ultimo
passaggio conteso. Un giudice sensibile alla lunghezza, alla sicurezza del
tono o alla ripetizione (gli stessi bias documentati per l'*LLM-as-a-judge*
nella sezione su LLMOps) rompe l'ipotesi, e il gioco premia la persuasione
invece della verità. La seconda riguarda proprio questo capitolo: due
dibattenti istanziati dallo **stesso modello** ereditano la correlazione della
sezione precedente. Se entrambi condividono lo stesso errore sistematico,
nessuno dei due lo contesta, l'errore non entra mai nel dibattito e il giudice
non ha nulla su cui esercitare la propria verifica. Il dibattito rende
*ispezionabile* il disaccordo che c'è, non crea quello che manca.

`````

## Quando qualcuno mente: i generali bizantini

Resta il caso duro, e ha un teorema che dice esattamente quanto è duro.
Nel 1982 Leslie Lamport, Robert Shostak e Marshall Pease pubblicano un
articolo destinato a diventare un classico dei sistemi distribuiti
{cite}`lamport1982byzantine`. La storiella è nota: alcune divisioni assediano
una città, i generali comunicano solo con messaggeri, devono decidere insieme
se attaccare o ritirarsi, e alcuni di loro sono traditori che possono dire
cose diverse a interlocutori diversi. Il nome ha una piccola storia editoriale
che Lamport ha raccontato lui stesso: la prima versione parlava di generali
**albanesi**, scelti perché l'Albania di allora era un paese chiuso e sembrava
improbabile che qualcuno protestasse; fu un collega, Jack Goldberg, a fargli
notare che di albanesi ce n'è anche fuori dall'Albania, e allora si ripiegò su
un impero abbastanza estinto da non offendersi.

`````{tab} Elementare

Facciamo il caso più piccolo: tre generali, di cui uno traditore, e messaggi
solo a voce. Uno comanda e gli altri due eseguono.

Prendi il punto di vista di un generale leale che riceve «attacca» dal
comandante. Chiede conferma al collega e quello risponde: «a me ha detto
ritirati». Adesso il nostro generale sa che uno dei due mente, ma non ha alcun
modo di sapere quale: le due storie sono perfettamente simmetriche. Potrebbe
essere il comandante che ha dato ordini diversi ai due luogotenenti, oppure il
collega che riferisce il falso. Dall'interno, le due situazioni sono
indistinguibili, perché sono fatte esattamente degli stessi messaggi.

Da qui il risultato: con soli messaggi a voce non basta la maggioranza dei
partecipanti onesti, ne serve più di **due terzi**. Per sopportare un solo
bugiardo servono almeno quattro partecipanti, per due almeno sette, per tre
almeno dieci. Non è che non si è ancora trovato l'algoritmo giusto: è
dimostrato che non esiste.

`````

`````{tab} Superiore

Formalmente, un comandante deve trasmettere un ordine a $n-1$ luogotenenti in
modo che valgano due condizioni: tutti i luogotenenti **leali** eseguono lo
stesso ordine, e se il comandante è leale ogni luogotenente leale esegue
proprio l'ordine impartito. Con messaggi **orali** (consegna garantita,
mittente identificabile, assenza di messaggio rilevabile, ma nessuna firma) il
risultato di Lamport, Shostak e Pease è di **impossibilità**: nessuna
soluzione esiste se i partecipanti sono $n \le 3f$, dove $f$ è il numero di
partecipanti che possono comportarsi in modo arbitrario. Serve dunque

$$
n \ge 3f + 1,
$$

e la condizione è anche sufficiente: l'algoritmo $\mathrm{OM}(f)$ degli autori
risolve il problema con $n \ge 3f+1$ in $f+1$ giri di messaggi, al prezzo di
un numero di messaggi che cresce come $O(n^{f+1})$, perché a ogni giro ciascun
luogotenente rigira a tutti gli altri quello che ha appena sentito. Cambiando
ipotesi cambia il limite: con **messaggi firmati**, cioè con firme non
falsificabili e verificabili da chiunque, il vincolo dei due terzi cade e
l'algoritmo $\mathrm{SM}(f)$ risolve il problema per qualunque $f$ purché
$n \ge f+2$ (sotto quella soglia i luogotenenti leali sono al più uno, e le due
condizioni valgono a vuoto). È un punto di metodo che vale ben oltre i
generali: un risultato di impossibilità non dice «impossibile», dice
«impossibile **sotto queste ipotesi**», e il lavoro dell'ingegnere è capire
quale ipotesi si può comprare.

`````

Che cosa ci fa un teorema sui protocolli di consenso in un libro di
intelligenza artificiale? Ci fa la distinzione che introduce, che è
esattamente quella che serve qui. Un partecipante **guasto** smette di
rispondere: se ne accorge chiunque, basta un timeout, e la contromisura è la
ridondanza (se uno tace, chiedi a un altro). Un partecipante **bizantino**
risponde, risponde in tempo, risponde in modo perfettamente plausibile, e dice
il falso; e non lo intercetta nessun timeout, perché dal punto di vista del
protocollo si sta comportando benissimo.

Un modello di linguaggio che allucina con sicurezza è, in questa
classificazione, un partecipante bizantino. Non si blocca, non restituisce un
errore, non abbassa il tono: produce una citazione inesistente con lo stesso
garbo con cui produce quelle vere. Ecco la ragione tecnica per cui la
**ridondanza ingenua non basta**: aggiungere copie protegge dai guasti, non
dalle bugie, e le architetture multi-agente costruite sull'idea «se sono in
tanti, qualcuno se ne accorgerà» stanno applicando la contromisura sbagliata al
guasto sbagliato. È lo stesso terreno del capitolo sull'**AI responsabile**,
dove la robustezza non è la capacità di non rompersi ma quella di comportarsi
in modo prevedibile quando qualcosa (o qualcuno) prova a farti sbagliare.

Va detto con onestà fin dove arriva l'analogia. Il teorema descrive un
avversario che sceglie la strategia peggiore possibile e può coordinarsi con
gli altri traditori: un modello che sbaglia non è malevolo in quel senso, e la
soglia $3f+1$ non si trasferisce come formula ai sistemi di agenti. Quello che
si trasferisce, e non è poco, è la classificazione dei guasti e la sua
conseguenza di progetto: contro un partecipante che mente con garbo, l'unica
difesa è un **riscontro esterno** che non passi per la sua parola (uno
strumento che verifica, una fonte consultabile, un test che gira), cioè il
cancello deterministico del loop engineering, qui in veste di antidoto al
bizantinismo. E vale, sempre, il conto della sezione precedente: un
verificatore che non ha modo di provare davvero non aggiunge informazione,
aggiunge una firma {cite}`cemri2025why`.

## Rendere visibile il disaccordo

C'è una cosa che il teorema dei generali bizantini garantisce e una che non
garantisce, e la distinzione chiude la sezione. Garantisce l'**accordo**: i
partecipanti leali decidono tutti lo stesso valore. Non garantisce la
**verità**: se il comandante leale ordina una ritirata sbagliata, il protocollo
farà ritirare tutti, ordinatamente e all'unanimità. Consenso e correttezza sono
due proprietà diverse, e confonderle è il modo più elegante di costruire un
sistema che sbaglia in modo coordinato.

Vale per tutti i meccanismi visti qui. I messaggi tipizzati non rendono un
agente più intelligente: rendono controllabile se ha risposto e se ha mantenuto
un impegno. Il voto non aggiunge competenza: amplifica quella che c'è, nel bene
e nel male. Il dibattito non produce argomenti veri: rende esaminabile quello
che i due dibattenti hanno scelto di contestare. Nessuno di questi protocolli
rende affidabili gli agenti; tutti rendono **osservabile il loro disaccordo**.

Ed è già molto, purché si progetti per quello. In pratica significa tre cose:
registrare i voti e non solo il vincitore (un $5$ a $4$ e un $9$ a $0$ hanno lo
stesso esito e valgono in modo diversissimo); trattare il disaccordo come un
segnale da instradare, verso un giudice, un altro strumento o una persona,
invece che come rumore da appiattire; e diffidare dell'unanimità almeno quanto
del conflitto, perché in un gruppo di agenti costruiti tutti allo stesso modo
il consenso perfetto è più spesso la firma di un errore condiviso che la prova
di una risposta giusta. Un sistema in cui il disaccordo emerge ed è
ispezionabile è migliore di uno che converge in silenzio sulla risposta
sbagliata.

```{admonition} Da ricordare
:class: important
- Un messaggio fra agenti ha un **contenuto** e una **forza illocutoria** (che
  cosa fa: chiedere, informare, impegnarsi, rifiutare). KQML e FIPA-ACL l'hanno
  resa un campo obbligatorio, e tipizzare i messaggi trasforma la conversazione
  in una **macchina a stati ispezionabile**: si può verificare a macchina se una
  richiesta ha avuto risposta e se un impegno è stato onorato. Il **Contract
  Net** {cite}`smith1980contract` (bando → offerta → aggiudicazione) è il
  capostipite di questi protocolli.
- Il **teorema di Condorcet**: con $n$ votanti **indipendenti** corretti con
  probabilità $p > 1/2$, la maggioranza tende alla verità
  ($p = 0{,}7$: $P_3 = 0{,}784$, $P_9 = 0{,}901$). Sotto $1/2$ converge
  all'errore: il voto amplifica la tendenza di fondo, non aggiunge competenza.
- **L'ipotesi crolla fra agenti identici.** Dieci istanze dello stesso modello
  sono un votante interrogato dieci volte: con una frazione $\lambda$ di errori
  sistematici il voto ha un tetto $1-\lambda$ ($0{,}80$ contro l'$1$ promesso),
  e con correlazione $\rho$ il numero efficace di votanti si ferma a $1/\rho$.
  Il voto non aumenta la correttezza, aumenta la **confidenza**: su nove agenti
  concordi, il 45% delle unanimità è sbagliato.
- Un po' di indipendenza si compra: campionare a temperatura, imporre percorsi
  di ragionamento diversi, riformulare la domanda, cambiare modello. È il
  motivo per cui la **self-consistency** {cite}`wang2023selfconsistency`
  funziona, ma decorrela solo **parzialmente**: il guadagno reale è sempre
  minore di quello che Condorcet promette.
- Il **dibattito** {cite}`irving2018ai` sfrutta l'asimmetria fra produrre e
  verificare (con giudice polinomiale: da NP a PSPACE), ma regge solo se il
  giudice riconosce un argomento fallace, e due dibattenti dello stesso modello
  ereditano la stessa correlazione.
- I **generali bizantini** {cite}`lamport1982byzantine`: con soli messaggi
  orali servono $n \ge 3f+1$ partecipanti per tollerarne $f$ che mentono (con
  firme il vincolo cade). Un agente **guasto** tace e lo becca un timeout; uno
  **bizantino** risponde in modo plausibile e falso, come un LLM che allucina
  con sicurezza: contro di lui la ridondanza non serve, serve un riscontro
  esterno. E il consenso garantisce l'**accordo**, mai la **verità**.
```
