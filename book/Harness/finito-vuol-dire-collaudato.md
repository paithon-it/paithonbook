# Finito vuol dire collaudato: la verifica di un lavoro lungo

Rajasekaran parte da una tendenza che scrive in chiaro: chiesto di giudicare il
lavoro che ha prodotto, un agente tende a lodarlo con sicurezza, anche quando a
una persona la qualità sembra chiaramente mediocre {cite}`rajasekaran2026harness`.
La sua risposta è stata separare i ruoli, un agente che costruisce e uno che
valuta. Non perché il valutatore separato fosse imparziale per natura: è un
modello anche lui, incline alla generosità verso il testo di un modello, e
appena messo al lavoro trovava problemi veri, poi si convinceva da solo che non
fossero gravi, e approvava; tendeva a provare l'applicazione in superficie,
senza spingersi nei casi limite (un carrello vuoto, un prezzo a zero, un nome
lunghissimo), e i difetti più sottili spesso passavano. Ma rendere scettico un
valutatore separato si è rivelato molto più trattabile che rendere critico chi
costruisce verso il proprio lavoro; e una volta che il giudizio esterno c'è,
chi costruisce ha qualcosa di concreto su cui correggersi.

Lo stesso guasto ha già dato una regola al {doc}`loop engineering
</IngegneriaLLM/loop-engineering>`: chi propone non può essere anche chi
approva, e un cancello di verifica la fa rispettare. Su un lavoro lungo, però,
il problema cambia forma in due punti. Il primo è *che cosa* si verifica: una
funzionalità di un'applicazione vera non è una funzione del codice (un pezzetto
di programma che fa un calcolo) con i suoi test, ma un pezzo che deve
funzionare insieme agli altri, come lo userebbe una persona. Il secondo è *nel
tempo*: in un lavoro che dura centinaia di sessioni, chi scrive «fatto»
sull'elenco, e chi controlla il controllore, diventano scelte da fare nel
progettare l'harness. Fra i guasti dell'esperimento di Young ce n'erano due di
questa famiglia, la vittoria dichiarata da una sessione che vedeva molto codice
scritto, e le funzionalità segnate come finite dopo averle provate soltanto a
pezzi.

## Il collaudo da utente

Nell'esperimento di Young l'agente segnava le funzionalità come finite dopo
averle controllate con i test d'unità (le prove automatiche che verificano un
pezzo di codice da solo) o con una richiesta mandata direttamente al server, il
programma che sta dietro le pagine, senza mai provarle da un capo all'altro
{cite}`young2025harnesses`. La riparazione è stata dargli gli strumenti per
usare l'applicazione come un utente, cioè per guidare un browser: aprire le
pagine, premere i pulsanti, leggere quello che compare. Con l'istruzione
esplicita di usarli, l'agente verificava per lo più bene le funzionalità da un
capo all'altro, e trovava difetti che dal codice non si vedevano; Young scrive
che questi strumenti hanno migliorato nettamente i risultati. Lo stesso
articolo nomina il limite: il browser automatico non vedeva i riquadri d'avviso
del browser stesso, le finestrelle del tipo «Sei sicuro? OK, Annulla», e le
funzionalità che ne dipendevano erano, di conseguenza, più difettose delle
altre. Il collaudo vede soltanto quello che i suoi strumenti gli fanno vedere.

Un test d'unità isola il pezzo che prova, e quando il pezzo ha bisogno di un
altro pezzo lo sostituisce con una controfigura, scritta per comportarsi come
il test si aspetta. Il confine fra i pezzi, il test lo vede soltanto attraverso
la controfigura: se chi l'ha scritta condivide la convinzione del pezzo che si
prova, il difetto di confine passa, e succede facilmente quando pezzo e
controfigura nascono nella stessa sessione. I difetti di confine hanno qualche
forma ricorrente. Due componenti si scambiano un dato con convenzioni diverse,
un formato o un'unità di misura (centesimi contro euro), o un percorso di file
scritto in un modo dove l'altro lo vuole in un altro. Un componente cambia uno
stato, cioè un dato tenuto in memoria, e un altro continua a leggerlo vecchio.
Una risorsa, come un file o una connessione a un database, viene aperta da uno
e mai chiusa, finché quelle disponibili non finiscono. E il comportamento
cambia dall'ambiente di prova a quello vero. Il primo caso sta in un programma,
nel negozio online del {doc}`repository come memoria
<il-repository-come-memoria>`: un carrello e la sua pagina, scritti in due
sessioni diverse, ciascuno con il suo test che passa; poi il collaudo che li
monta insieme, e la voce dell'elenco che si chiude soltanto con il suo esito.

```python
# Due componenti del negozio, scritti in due sessioni diverse.
class Carrello:
    """Tiene i prezzi in centesimi, come si fa per non perdere spiccioli."""
    def __init__(self):
        self.righe = []
    def aggiungi(self, nome, prezzo_euro):
        self.righe.append((nome, round(prezzo_euro * 100)))
    def totale(self):
        return sum(centesimi for _, centesimi in self.righe)

def pagina_totale(carrello):
    """La pagina del carrello: chi l'ha scritta credeva il totale in euro."""
    return f"Totale: {carrello.totale():.2f} €".replace(".", ",")

# I test d'unità: ciascuno prova il suo pezzo da solo.
def test_carrello():
    c = Carrello()
    c.aggiungi("tazza", 19.99)
    return c.totale() == 1999                     # centesimi: giusto

class CarrelloFinto:                              # la controfigura del test
    def totale(self):                             # della pagina: risponde in
        return 19.99                              # euro, come la pagina crede

def test_pagina():
    return pagina_totale(CarrelloFinto()) == "Totale: 19,99 €"

# Il collaudo della voce: i pezzi veri, montati insieme, e i suoi passi di
# prova. Qui si legge lo schermo come una stringa; un harness vero passa da
# un browser.
def collaudo_carrello():
    c = Carrello()
    c.aggiungi("tazza", 19.99)                    # «aggiungi una tazza»
    schermo = pagina_totale(c)
    return schermo == "Totale: 19,99 €", schermo  # «controlla il totale»

voce = {"description": "Un utente aggiunge un prodotto e vede il totale",
        "steps": ["Aggiungi una tazza da 19,99 €",
                  "Controlla che il totale sia il prezzo del prodotto"],
        "passes": False}
collaudi = {voce["description"]: collaudo_carrello}   # un collaudo per voce

def chiudi_voce(voce, dichiarazione_agente):
    """Il campo `passes` lo scrive il collaudo della voce; la dichiarazione
    dell'agente non entra nel conto."""
    ok, schermo = collaudi[voce["description"]]()
    voce["passes"] = ok
    return ok, schermo

print("test d'unità del carrello:", "passa" if test_carrello() else "fallisce")
print("test d'unità della pagina:", "passa" if test_pagina() else "fallisce")
ok, schermo = chiudi_voce(voce, dichiarazione_agente="fatto, i test passano")
print("collaudo da utente:", "passa" if ok else "fallisce",
      f"(lo schermo dice «{schermo}»)")
print("passes:", voce["passes"])
```

```text
test d'unità del carrello: passa
test d'unità della pagina: passa
collaudo da utente: fallisce (lo schermo dice «Totale: 1999,00 €»)
passes: False
```

I due test sono giusti, ciascuno dal suo punto di vista, ed è proprio questo il
guaio: il carrello conta in centesimi, la pagina crede di ricevere euro, e la
controfigura del carrello usata nel test della pagina ha preso per buona la
convinzione della pagina. Solo montando i due pezzi veri e guardando lo
schermo, come farebbe chi compra, la tazza costa millenovecentonovantanove
euro. La funzione che chiude la voce mostra l'altra metà della lezione: il
campo `passes` lo scrive l'esito del collaudo di quella voce, e la
dichiarazione dell'agente («fatto, i test passano») non entra nemmeno nel
conto. Nell'harness di Young il segno lo metteva l'agente, ma solo dopo la
prova da utente; farlo scrivere al codice del collaudo è la versione che non
dipende dalla buona volontà di nessuno.

`````{tab} Elementare

La biblioteca ordina una libreria con le ante, e il disegno dice soltanto
questo. L'artigiano di lunedì fa la cassa, cioè il mobile senza le ante, e fora
i buchi per le cerniere, i perni su cui l'anta gira, sul lato sinistro, come si
fa da sempre in quella bottega. L'artigiano di martedì fa le ante, e siccome
dove ha imparato il mestiere le ante si aprono a destra, le fa per aprirsi a
destra. Per provarle si costruisce una cassa finta di cartone, con i buchi
dove pensa lui, a destra: le ante ci stanno benissimo. Martedì sera annota
«fatto» sul registro, sicurissimo.

Il collaudo vero è un altro. Si monta la libreria, la si mette in piedi come
starà in biblioteca, e si provano le ante come le userà chi ci va a prendere un
libro. Le ante non si riescono nemmeno ad attaccare: i buchi delle cerniere
sono dall'altra parte. Nessuno dei due pezzi era sbagliato rispetto al disegno;
era sbagliato il punto in cui si toccano, e la cassa finta, fatta con le idee
di chi provava le ante, non poteva accorgersene.

Anche chi collauda, cioè chi prova il mobile montato prima della consegna, ha i
suoi limiti, e vanno saputi. Apre le ante e guarda, ma se la serratura si prova
con una chiave che a lui non hanno dato, la serratura non la prova: è lì che
finiranno i difetti. E la casella «fatto» sul foglio dell'ordine si spunta
soltanto con il collaudo passato: la parola dell'artigiano non basta. Meglio
ancora se a spuntarla è il collaudo stesso, così non dipende dalla buona
volontà di nessuno.

Il collaudo intero, però, costa: montare la libreria e provarla richiede
mezz'ora, misurare un pezzo un minuto. Per questo la bottega fa prima le prove
rapide che sbagliano spesso, e poi quelle lunghe; ma se si salta il montaggio
ogni volta che «tanto non tocca le ante», si salta proprio quando un artigiano
è convinto di non aver toccato un confine, cioè quando ci sono i difetti di
confine.

`````

`````{tab} Superiore

Sia una funzionalità realizzata da componenti $u_1, \dots, u_n$, con
interfacce $\iota_{ij}$ fra le coppie $(i, j) \in I$ che si parlano, e sia
$V_i$ il suo collaudo, i passi di prova della voce, osservabili dall'utente.
Un test d'unità di $u_i$ controlla, su pochi ingressi, una proprietà
$\psi_i(u_i; \hat{u})$ di $u_i$ montato con le controfigure $\hat{u}_j$, cioè
con modelli del comportamento dei componenti con cui comunica. I test danno il
successo dell'utente soltanto se valgono tre ipotesi che i test non
controllano: che $\psi_i$ valga su tutti gli ingressi che contano, e non solo
su quelli provati; che ogni controfigura si comporti come il componente vero
sull'interfaccia che li lega, così che $\psi_i(u_i; \hat{u})$ dica qualcosa di
$\psi_i(u_i; u)$; e che la scomposizione sia giusta, cioè che le $\psi_i$ sui
componenti veri implichino il collaudo (il passo del ragionamento
*assume-guarantee*):

$$
\bigwedge_{i} \psi_i(u_i; \hat{u}) \;\wedge\;
\bigwedge_{(i,j) \in I} \big(\hat{u}_j \simeq u_j \text{ su } \iota_{ij}\big)
\;\wedge\; \Big(\bigwedge_i \psi_i(u_i; u) \Rightarrow V_i = 1\Big)
\;\Longrightarrow\; V_i = 1 ,
$$

con le $\psi_i$ intese su tutto il dominio d'ingresso. Nell'esempio del
carrello le $\psi$ valgono per entrambi i componenti con le loro controfigure,
mentre $\hat{u} \simeq u$ è falsa sull'unità di misura del totale. Il collaudo
da capo a capo valuta $V_i$ sui componenti montati, e toglie l'ipotesi di
fedeltà per tutto ciò che monta davvero; la conserva per ciò che resta
simulato, l'ambiente di prova e i servizi esterni finti, ed è lì che vive la
dipendenza dall'ambiente. Fra i due estremi stanno i test che controllano la
fedeltà direttamente, montando due componenti veri o confrontando una
controfigura con il componente che imita (i test di contratto). Le famiglie
dei difetti di confine corrispondono ai modi in cui $\hat{u}_j \simeq u_j$
fallisce: formato o unità dei dati, stato condiviso, ciclo di vita delle
risorse, ambiente.

Il collaudo da utente ha un limite di osservabilità e uno di costo. $V_i$ si
valuta su ciò che gli strumenti del collaudo espongono, e ciò che non
espongono resta non verificato (i riquadri d'avviso nativi del browser,
invisibili all'automazione usata da Young). E un collaudo da capo a capo è più
lento e più fragile di un test d'unità. Se tutti i controlli girano sempre,
l'ordine decide solo quanto presto si scopre un guasto: con l'arresto al primo
fallimento e controlli indipendenti, il costo atteso è minimo ordinandoli per
$c_i / p_i$ crescente, il costo diviso la probabilità di fallire, non per
costo. Se invece il collaudo intero gira solo quando la modifica «tocca un
confine», l'ordine cambia anche ciò che si verifica, e la decisione ha falsi
negativi proprio sui difetti di confine: chi ha scritto la pagina del carrello
non credeva di toccarne uno. Per questo la routine di Young fa una prova da
utente a ogni sessione.

`````

## Messaggi che insegnano, regole che si promuovono

Un cancello che dice soltanto «no» costringe chi è stato respinto a indovinare
perché. Il loop engineering ne ha già fatto una regola, il motivo del rifiuto
passato al giro successivo; nel repository di Lopopolo la stessa idea è scesa
fino ai controlli automatici del codice. La squadra usa dei linter propri,
cioè programmi che leggono il codice e segnalano ciò che viola una regola,
generati anch'essi dall'agente; e siccome sono fatti su misura, i loro
messaggi d'errore si scrivono come si vuole, e la squadra ci ha messo dentro le
istruzioni per riparare: il messaggio entra nel contesto dell'agente, e
l'agente sa che cosa fare {cite}`lopopolo2026harness`.

La disciplina poggia su due principi. Il primo è far rispettare delle
invarianti, cioè delle proprietà che devono valere sempre, senza prescrivere
come si scrive il codice: l'agente deve controllare la forma dei dati nel punto
in cui entrano nel sistema, ma la libreria con cui farlo la sceglie lui. Il
codice di ogni area del prodotto, poi, è diviso in strati fissi, dai tipi dei
dati, in fondo, fino all'interfaccia che l'utente vede, in cima, e ogni strato
può usare soltanto quelli che lo precedono, mai quelli dopo; un controllo
automatico rifiuta tutto il resto, così che una modifica all'interfaccia non
possa rompere il modo in cui i dati sono definiti. Il secondo principio è che
il gusto delle persone entra nel sistema una volta e poi vale sempre: i
commenti delle revisioni, le richieste di modifica per riordinare il codice, i
difetti visti dagli utenti diventano documentazione o controlli automatici, e
quando la documentazione non basta la regola viene promossa a codice, un
controllo che fallisce invece di una frase che si può dimenticare.

`````{tab} Elementare

Il collaudatore che respinge la libreria non scrive soltanto «no» sul
cartellino. Scrive: «i buchi delle cerniere sono a sinistra, le ante aprono a
destra; per le ante a destra si usa la sagoma B, secondo cassetto».
L'artigiano di domani legge il cartellino e sa che cosa rifare, senza dover
capire da solo dove sta il problema.

Poi, se lo stesso errore torna tre volte, il proprietario lo scrive sul
cartello all'ingresso. E se torna anche dopo, smette di scriverlo e costruisce
una maschera di foratura: una sagoma che si appoggia sulla cassa finita e ha i
fori soltanto dove vanno. Un buco dalla parte sbagliata la maschera non lo
copre, e il pezzo torna indietro subito, al banco, con il biglietto che dice
quale maschera usare. Una regola scritta si può non leggere; una maschera che
respinge il pezzo no.

Il proprietario però non pretende di dire come si tiene il trapano o quale
punta usare: quello lo decide l'artigiano. Pretende soltanto le cose che
contano: che i buchi stiano dove devono stare, che il legno entri in bottega
solo dopo essere stato controllato alla porta, e che i lavori vadano in un
ordine fisso, prima il taglio, poi le parti, poi il montaggio, poi la vernice.
Chi monta può prendere le parti, ma chi taglia non va mai a prendere un pezzo
dal reparto vernici, e un controllo all'uscita di ogni reparto lo impedisce.

`````

`````{tab} Superiore

Un controllo automatico è una verifica deterministica sul codice, e il
messaggio che accompagna il rifiuto è l'osservazione $o_{k,t}$ che il ciclo
rimette nel contesto. Il suo valore per il passo successivo dipende da quanta
informazione porta sulla riparazione. Un rifiuto binario porta al più un bit:
dice soltanto che lo stato corrente è fuori dall'insieme ammesso. Un messaggio
che nomina una regola fra $R$ e un punto fra $n$ ne porta fino a
$\log_2(R\,n)$, e se aggiunge la correzione sposta la distribuzione
$P_\theta(\cdot \mid C)$ dell'azione successiva verso le azioni che riparano.
È la differenza fra un oracolo che risponde sì o no e un oracolo che
restituisce un controesempio con la sua spiegazione.

Le invarianti di architettura sono vincoli sul grafo delle dipendenze. Sia
$\mathcal{G} = (N, A)$ il grafo dei moduli, con un arco $(x, y) \in A$ quando
$y$ usa $x$, e sia $h: N \to \{1, \dots, S\}$ lo strato di ciascuno, dai tipi
($1$) all'interfaccia ($S$). La regola «dipendenze solo in avanti» chiede
$h(x) \le h(y)$ per ogni arco, salvo quelli che passano da un'interfaccia
esplicita per le funzioni trasversali; nel repository di Lopopolo gli strati
sono tipi, configurazione, accesso ai dati, servizi, esecuzione e interfaccia,
e le funzioni trasversali (autenticazione, connettori, telemetria) entrano da
un'unica interfaccia {cite}`lopopolo2026harness`. La fonte parla di un insieme
limitato di archi ammessi, quindi la regola vera è più stretta della
disuguaglianza, che ammetterebbe ogni arco in avanti e i cicli dentro uno
strato: è un sottoinsieme fissato delle coppie $(h(x), h(y))$. Si verifica in
tempo lineare nella dimensione del grafo, se le dipendenze si estraggono
staticamente dal codice, e non dice niente su come è scritto ciascun modulo:
vincola la struttura, non l'implementazione. La promozione di una regola dalla
documentazione al codice trasforma un vincolo che l'agente rispetta con una
certa probabilità, se lo carica e lo segue, in un predicato che ogni modifica
deve soddisfare per entrare.

`````

## Chi può toccare il metro

Un cancello misura il lavoro con un metro, e il metro deve stare fuori dalla
portata di chi viene misurato. Le istruzioni di Young lo dicono per le voci
dell'elenco, che lui chiama test: toglierle o modificarle è inaccettabile,
perché potrebbe portare a funzionalità mancanti o difettose
{cite}`young2025harnesses`. Il loop engineering lo aveva visto dall'altro lato:
se il ciclo permette al modello di modificare i test del codice, il cancello
misura solo la coerenza del modello con sé stesso.

Un esperimento di ricerca mostra quanto la partizione sia facile da dichiarare
e difficile da ottenere. Nel marzo 2026 Andrej Karpathy, uno dei ricercatori
più noti dell'apprendimento profondo, ha pubblicato `autoresearch`, un piccolo
impianto in cui un agente fa esperimenti per conto suo, di notte,
sull'addestramento di un modello di linguaggio in miniatura, un piccolo
programma che impara a prevedere il pezzo di testo successivo
{cite}`karpathy2026autoresearch`. I file sono tre, e il progetto sta in chi può
toccare quale. `prepare.py` contiene i dati, il modo di leggerli e la funzione
di valutazione, e le istruzioni dicono che è in sola lettura; `train.py`
contiene il modello, l'ottimizzatore e il ciclo di addestramento, ed è l'unico
file che l'agente modifica; `program.md` contiene le istruzioni per l'agente, e
lo scrive la persona. Ogni esperimento dura cinque minuti di addestramento, così
che un modello più grande ma più lento non vinca soltanto perché ha avuto più
tempo, e si giudica su una sola grandezza, la `val_bpb`: quanti bit servono in
media per ogni carattere di un testo che il modello non ha visto durante
l'addestramento (più il modello prevede bene, meno ne servono). Le istruzioni
dicono di tenere un cambiamento se la misura scende e di scartarlo altrimenti,
con un criterio di semplicità: un guadagno minimo con codice molto più semplice
si tiene, uno minimo con venti righe contorte no. A decidere, e a tornare
indietro, è l'agente stesso.

La partizione, però, è dichiarata più che imposta. Che `prepare.py` resti
intatto lo chiede un'istruzione, e nel repository non c'è un controllo che lo
verifichi. Ed è meno netta di come sembra: la funzione di valutazione sta in
`prepare.py`, ma per calcolare la perdita chiama il modello definito in
`train.py`, e anche il ciclo che misura i cinque minuti e la riga che stampa la
misura stanno in `train.py`. Fissi, fuori dalla portata dell'agente, restano i
dati di validazione e il conto che trasforma la perdita in bit per carattere;
la perdita di ogni token la calcola il file che l'agente scrive.

`````{tab} Elementare

La dima di collaudo si usa ma non si tocca: è fissata al banco del collaudo, e
la chiave per smontarla ce l'ha il proprietario. L'artigiano può provarci il
pezzo cento volte, ma non può limarla perché il pezzo ci entri. Se potesse, la
dima direbbe sempre «va bene», e non servirebbe più a niente.

Una bottega può anche fare esperimenti di notte da sola, e funzionano solo se
si rispetta questa regola. Il proprietario lascia scritto che cosa provare, per
esempio una vernice più resistente; l'artigiano di turno cambia la ricetta
della vernice (è l'unica cosa che può toccare), la stende su una tavoletta,
aspetta un'ora esatta e la gratta con la macchina di prova. Se la vernice
nuova resiste più di quella di ieri, la ricetta nuova resta; se no, si torna
alla ricetta di ieri. Sempre un'ora, sempre la stessa macchina: così le prove
della notte si confrontano fra loro.

Due cose, però, possono tradirla. Se la macchina di prova sta sul banco
dell'artigiano, e il cartello «non toccare» è l'unica cosa che la protegge,
basta una notte storta perché venga regolata; e se la punta che gratta la
vernice la sceglie lui insieme alla ricetta, la prova non è più del tutto del
proprietario. E una sola grattata su una tavoletta può ingannare: due
tavolette con la stessa vernice non resistono mai esattamente uguale, e una
ricetta può vincere per caso. A forza di tenere le ricette che vincono per un
soffio, sempre sulla stessa tavoletta, la bottega rischia di ritrovarsi con
una vernice scelta su misura per quella tavoletta, e non migliore per i mobili.

`````

`````{tab} Superiore

Sia $\mathcal{W}$ l'insieme dei file che l'agente può scrivere e $\mathcal{V}$
quello dei file che definiscono la verifica, escluso l'oggetto verificato (i
collaudi, l'elenco, il codice di valutazione, i dati di prova). La condizione è
$\mathcal{W} \cap \mathcal{V} = \varnothing$: se l'intersezione non è vuota,
l'agente può aumentare il valore del predicato di successo cambiando il
predicato invece del prodotto. È la manomissione dello strumento di misura, il
*reward tampering* che il {doc}`primo turno <il-primo-turno>` distingue dalla
legge di Goodhart, la quale resta anche con l'intersezione vuota: il
cricchetto descritto più sotto ne è un esempio, con l'adattamento ai dati di
validazione. Esclusa dalla definizione deve restare la sola parte del calcolo
che è l'oggetto stesso, e qui `autoresearch` è istruttivo: il modello misurato
calcola una parte della propria misura. Dichiarati, $\mathcal{W} =
\{\texttt{train.py}\}$ e $\mathcal{V} \subseteq \{\texttt{prepare.py}\}$;
effettivi, la perdita per token (il `forward` con l'entropia incrociata), il
budget di tempo e la stampa del risultato stanno in $\mathcal{W}$, e la
separazione da $\mathcal{V}$ la difende un'istruzione. Per renderla una
garanzia servirebbero la perdita calcolata fuori dal modello, sui suoi soli
logit, il tempo misurato da un processo esterno, e la misura calcolata dal
ciclo invece che stampata dal file che l'agente scrive.

Lo schema dell'esperimento è un *ratchet*, un cricchetto: idealizzando la
regola di `program.md`, si accetta una modifica se $\text{bpb}(\text{nuovo}) <
\text{bpb}(\text{corrente})$, altrimenti si torna allo stato precedente; nella
regola vera entrano anche il criterio di semplicità e la memoria della scheda
grafica, che resta un vincolo morbido, e la decisione la prende l'agente. Il
budget fisso per esperimento, cinque minuti di orologio, rende la misura
confrontabile fra esperimenti che cambiano l'architettura o l'ottimizzatore: si
confrontano modelli allenati con la stessa spesa, non con lo stesso numero di
passi. Lo paga in rumore, perché il numero di passi in cinque minuti cambia con
il carico della macchina anche a seme fisso. Il cricchetto ha i suoi limiti
noti: è una ricerca locale e avida, che non accetta un peggioramento
temporaneo; su una misura rumorosa accetta miglioramenti dovuti al caso; e il
valore corrente, scelto come il minimo di molte estrazioni rumorose sullo
stesso frammento di validazione, è distorto verso il basso, cioè la ricerca si
adatta ai dati di validazione man mano che sceglie.

`````

## Chi controlla il controllore

Il valutatore di Rajasekaran era un modello, e per farlo funzionare è servita
una taratura. Il metodo era semplice e faticoso: leggere i resoconti del
valutatore, trovare i punti in cui il suo giudizio si allontanava da quello
dell'autore, e riscrivere le istruzioni per correggere quel genere di errore.
Dopo qualche giro il valutatore giudicava in un modo che l'autore trovava
ragionevole, e qualche difetto restava comunque scoperto
{cite}`rajasekaran2026harness`. Per le valutazioni di gusto (il disegno di
un'interfaccia, cioè delle schermate che l'utente vede) l'autore ha aggiunto
alle istruzioni degli esempi con il punteggio spiegato voce per voce.

Il valutatore giudicava ogni tappa del lavoro su quattro criteri fissi: quanto
il prodotto fosse completo, se funzionasse, come fosse disegnato, com'era
scritto il codice. Ciascuno aveva una soglia, e una sola soglia mancata bastava
a respingere la tappa. Prima di ogni tappa, poi, chi costruisce e chi valuta
stipulavano un **contratto**: chi costruisce proponeva che cosa avrebbe
consegnato e con quali comportamenti verificabili lo si sarebbe provato, chi
valuta rivedeva la proposta, e i due ripetevano finché non erano d'accordo,
scambiandosi dei file. Per una tappa sola i criteri del contratto erano
ventisette, tutti sull'editor dei livelli di un videogioco. Rajasekaran lo
motiva con lo spazio fra una richiesta scritta in termini generali e una
consegna che si possa controllare: il contratto decide prima, e per iscritto,
che cosa vorrà dire «fatto».

`````{tab} Elementare

Il proprietario assume un collaudatore nuovo, e i primi giorni il collaudatore
è troppo buono: vede che un'anta non chiude bene, la annota, e poi scrive
«piccolezza, approvato». Il proprietario legge i suoi cartellini la sera, uno
per uno, e segna dove lui avrebbe respinto il pezzo e il collaudatore no. Poi
gli riscrive le istruzioni: «un'anta che non chiude è un pezzo da rifare,
sempre», e gli lascia tre pezzi d'esempio con accanto il giudizio giusto e il
perché. Il collaudatore, come gli artigiani, la mattina non ricorda niente: a
migliorare non è lui, sono le istruzioni scritte e i pezzi d'esempio.

Per sapere se le istruzioni nuove funzionano, il proprietario non le prova sui
pezzi su cui le ha scritte: tiene da parte qualche pezzo che il collaudatore
non ha mai visto, e guarda come lo giudica. E sta attento anche all'errore
opposto, il collaudatore diventato troppo severo, che fa rifare pezzi buoni e
fa girare la bottega a vuoto. Dopo qualche settimana il collaudatore giudica in
modo più vicino al proprietario, ma non uguale: qualche difetto gli sfugge
ancora, e su duecento pezzi anche un collaudatore quasi sempre giusto ne lascia
passare qualcuno. E siccome gli artigiani, giorno dopo giorno, imparano che
cosa il collaudatore guarda, i difetti che gli sfuggono sono proprio quelli dei
pezzi fatti per piacergli.

Il collaudatore guarda sempre quattro cose, con un minimo per ciascuna: che il
mobile sia completo, che funzioni, che sia bello, che sia fatto bene dentro; se
una sola è sotto il minimo, il pezzo torna indietro. E prima di cominciare un
pezzo, artigiano e collaudatore si mettono d'accordo per iscritto su come lo si
proverà: «lo scaffale regge i volumi dell'enciclopedia, i più pesanti, per una
notte intera; le ante si chiudono da sole; il cassetto scorre fino in fondo».
Se il collaudatore dice che una notte è poco, si discute adesso, non a pezzo
finito. Quando il pezzo arriva, ogni prova del foglio ha un sì o un no, e basta
un no perché torni indietro.

`````

`````{tab} Superiore

Un valutatore che è a sua volta un modello è un classificatore binario sugli
esiti, con due tassi d'errore: $\alpha = P(\text{approva} \mid
\text{difettoso})$ e $\beta = P(\text{respinge} \mid \text{buono})$.
L'indulgenza descritta da Rajasekaran è un $\alpha$ alto. Per la sessione dopo
conta $P(\text{difettoso} \mid \text{approvato})$, che dipende anche dalla
frequenza dei difetti; e su un elenco lungo anche un $\alpha$ piccolo lascia,
fra le voci «fatte», qualche voce difettosa: la vittoria dichiarata per
accumulo. La taratura che Rajasekaran descrive è l'ottimizzazione di un prompt
su un insieme di casi etichettati da una persona, cioè un addestramento senza
gradiente, e ne eredita il rischio classico: un valutatore aggiustato sui casi
che si sono letti può adattarsi a quelli, e i suoi tassi vanno misurati su casi
tenuti da parte; misurati fuori dal ciclo, poi, possono sottostimare $\alpha$
dentro il ciclo, dove chi costruisce si corregge proprio contro quel
valutatore. I due errori hanno costi diversi. $\alpha$ lascia passare un
difetto, che la sessione dopo eredita come lavoro concluso. $\beta$ fa rifare un
lavoro buono: senza un tetto servono in media $1/(1 - \beta)$ tentativi, con un
tetto di $n$ la voce resta aperta con probabilità $\beta^n$, ed è una stima
ottimistica, perché gli errori dello stesso giudice sullo stesso lavoro sono
correlati. La {doc}`sezione sulle architetture degli agenti
</Agenti/architetture-e-valutazione>` discute lo stesso strumento, il modello
che giudica un modello, dal lato della valutazione, compresa la sua tendenza a
preferire i testi simili ai propri.

Le soglie e il contratto sono due oggetti diversi. I criteri fissi del
valutatore, con punteggi $\sigma_j$ e soglie $\tau_j$, danno la regola
congiuntiva $\text{accetta} \iff \forall j:\ \sigma_j \ge \tau_j$, che impedisce
a un criterio forte di compensarne uno debole ed è più severa di una media
pesata; ma con $K$ criteri e tassi $\beta_j$ indipendenti un lavoro buono viene
respinto con probabilità $1 - \prod_j (1 - \beta_j)$, che con ventisette
criteri binari e $\beta_j = 0{,}02$ fa circa $0{,}42$. Il contratto di tappa fissa
invece, prima della produzione, l'insieme dei comportamenti da provare,
ciascuno con esito sì o no. Che fissarlo prima aiuti anche contro l'indulgenza
è un ragionamento in più rispetto alla fonte, e vale per i criteri binari: lì
riconoscere il difetto equivale a respingere, e il valutatore non può
riconoscerlo e poi minimizzarlo. Su un criterio a punteggio l'indulgenza agisce
su $\sigma_j$, e una soglia fissata prima non la ferma.

`````

Collaudare da utente, far scrivere il segno «fatto» a chi collauda, tenere il
metro fuori dalla portata di chi viene misurato e tarare chi giudica sono le
difese contro la vittoria dichiarata. Passano tutte per gli strumenti che
l'agente ha in mano, il browser del collaudo come il comando che modifica un
file, e quegli strumenti si progettano come un’{doc}`interfaccia fra l'agente e
il computer <gli-strumenti>`.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Chi ha fatto un pezzo tende a dire «fatto» con sicurezza anche quando il
  pezzo è mediocre, e un collaudatore nuovo tende a essere troppo buono: per
  questo il segno «fatto» lo mette il collaudo, e il collaudatore si tara.
- Il collaudo vero monta i pezzi e li usa come li userà il cliente. Due pezzi
  giusti da soli possono essere sbagliati insieme, e la prova su un pezzo finto
  non lo vede, se il pezzo finto è fatto con le stesse idee di chi prova.
- Il collaudatore vede solo quello che i suoi strumenti gli fanno vedere: ciò
  che non può provare resta non provato.
- Un rifiuto spiega che cosa rifare. Un errore che torna si scrive sul
  cartello, e se torna ancora si costruisce una maschera che respinge il pezzo
  sbagliato. Il proprietario pretende le cose che contano, non il modo di
  tenere il trapano.
- La dima di collaudo si usa ma non si tocca. Negli esperimenti di notte
  l'artigiano tocca solo la ricetta, si prova sempre allo stesso modo, e si
  tiene il cambiamento solo se va meglio; ma una prova sola può vincere per
  caso, e un cartello «non toccare» non è una serratura.
- Il collaudatore si tara su pezzi che non ha mai visto, attenti a che non
  diventi né troppo buono né troppo severo; e prima di ogni pezzo ci si mette
  d'accordo per iscritto su come lo si proverà.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- L'autovalutazione è indulgente, e un valutatore separato si tara meglio di
  chi costruisce {cite}`rajasekaran2026harness`: `passes` lo scrive l'esito del
  collaudo, non la dichiarazione dell'agente.
- I test d'unità danno il collaudo solo con tre ipotesi: proprietà vere su
  tutto il dominio, controfigure fedeli sulle interfacce, scomposizione giusta.
  Il collaudo da capo a capo toglie la fedeltà per ciò che monta, non per ciò
  che resta simulato, ed è limitato da ciò che i suoi strumenti osservano
  {cite}`young2025harnesses`.
- Un messaggio di rifiuto porta fino a $\log_2(R\,n)$ bit invece di uno; le
  regole si promuovono dalla documentazione al codice, e le invarianti di
  architettura sono vincoli sul grafo delle dipendenze
  {cite}`lopopolo2026harness`.
- $\mathcal{W} \cap \mathcal{V} = \varnothing$, con $\mathcal{V}$ senza
  l'oggetto verificato; in `autoresearch` la partizione è dichiarata e
  difesa da un'istruzione {cite}`karpathy2026autoresearch`. Il cricchetto è
  una ricerca locale avida, esposta al rumore e all'adattamento ai dati di
  validazione.
- Un valutatore a modello ha $\alpha$ e $\beta$ da misurare su casi tenuti da
  parte; i criteri fissi con soglie danno una regola congiuntiva, il contratto
  fissa prima i comportamenti da provare.
```

`````
