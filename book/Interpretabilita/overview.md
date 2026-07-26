# Aprire la scatola nera

Nel 2016 tre ricercatori dell'Università di Washington — Marco Túlio Ribeiro,
Sameer Singh e Carlos Guestrin — addestrarono un classificatore a distinguere
le foto di **husky** da quelle di **lupo**. Sul set di prova il modello andava
benissimo, con un'accuratezza da fare invidia. Poi gli chiesero di *mostrare*
su cosa si basava, e la risposta fu imbarazzante: guardava la **neve**. Nelle
immagini di addestramento i lupi comparivano quasi sempre su sfondo innevato,
gli husky quasi mai; la rete aveva imparato una scorciatoia — «c'è neve →
lupo» — che con l'animale non c'entrava nulla {cite}`ribeiro2016why`. Un
rilevatore di neve travestito da riconoscitore di canidi.

La storia ha un antenato illustre. All'inizio del Novecento, a Berlino, un
cavallo di nome **Hans il Sapiente** sembrava saper contare: gli si chiedeva
«quanto fa sette più cinque?» e lui batteva lo zoccolo dodici volte. Nel 1907
lo psicologo Oskar Pfungst scoprì l'inganno: Hans non faceva aritmetica,
leggeva i movimenti involontari di chi poneva la domanda, che si irrigidiva
appena lo zoccolo raggiungeva il numero giusto. Bastava che l'esaminatore non
conoscesse la risposta, e Hans sbagliava. Da allora si chiama **effetto Clever
Hans** ogni sistema che *sembra* risolvere un problema mentre in realtà ne
risolve un altro, più facile e nascosto. Il classificatore di husky è un Clever
Hans in silicio: accuratissimo, e per la ragione sbagliata.

Il problema è che una rete neurale con milioni di parametri non ci dice, di suo,
*perché* decide come decide. È una **scatola nera**: entra un input, esce una
risposta, e in mezzo c'è un groviglio di numeri che nessuno legge a occhio. Fin
quando la posta in gioco è suggerire un film, poco male. Ma quando un modello
decide se concedere un mutuo, se un tumore è maligno o se rilasciare un
imputato, la domanda «perché?» diventa una questione di fiducia, di giustizia e
perfino di legge: il Regolamento generale sulla protezione dei dati europeo
(GDPR, in vigore dal 2018) ha introdotto norme sulle decisioni automatizzate e
un dibattuto «diritto alla spiegazione» per chi le subisce.

`````{tab} Elementare

Immagina un professore che dà sempre voti giusti ma non spiega mai come li
assegna. Finché i voti sono corretti ti fidi; ma il giorno che ne prendi uno
che ti sembra ingiusto, senza una spiegazione non puoi né capire dove hai
sbagliato né difenderti. E se scoprissi che il professore, invece di leggere il
compito, guarda di nascosto la calligrafia o il nome sul foglio? Sarebbe come
il cavallo Hans: risposte «giuste» ottenute con il trucco sbagliato.

Aprire la scatola nera vuol dire proprio questo: chiedere al modello non solo
*cosa* ha deciso, ma *su cosa* si è basato. Nel caso degli husky e dei lupi,
la spiegazione ha rivelato che il modello «vedeva» la neve e non il muso: un
errore che nessuna misura di accuratezza sul set di prova avrebbe mai
smascherato, perché anche nel test i lupi stavano sulla neve.

`````

`````{tab} Superiore

Il fenomeno degli husky ha un nome tecnico: **correlazione spuria** (o *shortcut
learning*). Il modello minimizza la sua *loss* sui dati disponibili, e se una
feature accessoria (la neve) è statisticamente associata all'etichetta nel
training *e* nel test, l'ottimizzazione la sfrutta senza scrupoli — è la
strategia più economica per abbassare l'errore. La metrica di generalizzazione
non lo cattura perché il bias è presente in entrambe le partizioni,
indistinguibili sotto l'ipotesi che siano campionate dalla stessa
distribuzione. È l'illusione dell'accuratezza: un modello «giusto per la
ragione sbagliata» collassa appena la distribuzione cambia — un lupo su erba,
un husky sulla neve — perché la scorciatoia appresa non è la relazione causale
che ci interessava. L'**interpretabilità** è lo strumento diagnostico che
espone la discrepanza tra ciò che il modello *dovrebbe* usare e ciò che *usa*
davvero, e che l'accuratezza aggregata, per costruzione, non può vedere.

`````

## Perché aprire la scatola

Le ragioni per volere una spiegazione non sono una sola, e non hanno tutte lo
stesso peso. Vale la pena elencarle, perché guidano *che tipo* di spiegazione
cerchiamo.

- **Fiducia.** Un medico non delega una diagnosi a un sistema di cui non capisce
  il ragionamento. La spiegazione è la condizione perché un esperto accetti di
  affidarsi al modello — o di scartarlo, come è giusto quando guarda la neve.
- **Debug.** Il caso husky è il manifesto: senza interpretabilità, un bug
  concettuale (la scorciatoia) resta invisibile dietro una buona accuratezza.
  Aprire la scatola è, prima di tutto, uno strumento di ingegneria.
- **Equità.** Un modello può discriminare per genere, etnia o codice postale
  anche senza che quelle variabili compaiano esplicitamente, riscoprendole da
  proxy correlati. Solo esaminando *su cosa* si basa una decisione si può
  scoprirlo — un tema che riprenderemo nel capitolo sull'AI responsabile.
- **Scoperta scientifica.** Quando un modello prevede la struttura di una
  proteina o l'attività di un farmaco, capire *cosa ha imparato* può suggerire
  ipotesi nuove ai ricercatori: il modello come microscopio, non solo come
  oracolo.
- **Obblighi normativi.** Dal credito all'assicurazione, un numero crescente di
  ordinamenti richiede che le decisioni automatizzate su una persona siano, in
  qualche misura, spiegabili e contestabili.

C'è un punto sottile che lega tutto: **la stessa decisione richiede spiegazioni
diverse a seconda di chi la riceve**.

`````{tab} Elementare

Chiedi «perché questo prestito è stato rifiutato?» a tre persone diverse e ti
aspetti tre risposte diverse. Al **cliente** serve una spiegazione azionabile:
«il reddito dichiarato è troppo basso rispetto alla rata; con una rata inferiore
la domanda passerebbe». All'**ingegnere** che ha costruito il modello serve
sapere quali variabili pesano e se ce n'è una sospetta (di nuovo: la neve). Al
**regolatore** serve la garanzia che il sistema non discrimini e che la
decisione sia contestabile. Una sola frase non può accontentarli tutti: la
spiegazione «buona» dipende da a chi parli e a cosa gli serve.

`````

`````{tab} Superiore

Doshi-Velez e Kim {cite}`doshi2017towards` insistono su questo: l'interpretabilità
non è una proprietà monolitica del modello, ma è relativa a un **compito a
valle** e a un **destinatario**. Ne deriva la loro tassonomia della
*valutazione* delle spiegazioni, su tre livelli di rigore crescente:
*application-grounded* (esperti reali sul compito reale — un medico che usa la
spiegazione in corsia), *human-grounded* (persone non esperte su compiti
semplificati, per esperimenti controllati), *functionally-grounded* (nessun
umano, ma una definizione formale di interpretabilità come proxy, per esempio
la profondità di un albero). La lezione è che «spiegabile» senza specificare
*per chi* e *per fare cosa* è un aggettivo vuoto: la stessa uscita del modello
va tradotta in linguaggi diversi per lo sviluppatore, l'utente finale e il
regolatore.

`````

## Una mappa delle spiegazioni

Il campo dell'*explainable AI* può sembrare una giungla di sigle. Tre assi,
ortogonali tra loro, mettono ordine e ci accompagneranno per tutto il capitolo.

`````{tab} Elementare

Pensa a tre domande da porre a ogni metodo di spiegazione.

- **Il modello è trasparente di suo, o va spiegato dopo?** Un piccolo albero di
  decisione — «se il reddito è sotto X e l'età sotto Y, rifiuta» — si legge come
  una ricetta: è **trasparente**. Una rete profonda no, e allora serve uno
  strumento esterno che la interroghi *dopo* l'addestramento (spiegazione
  **post-hoc**).
- **Vuoi capire tutto il modello, o una singola decisione?** «In generale questo
  modello dà molto peso al reddito» riguarda il modello **nel suo insieme**
  (spiegazione *globale*). «*Questo* prestito è stato rifiutato per via della
  rata troppo alta» riguarda **una risposta sola** (spiegazione *locale*).
- **Lo strumento funziona solo per un tipo di modello, o per qualunque
  modello?** Alcuni metodi guardano dentro un modello specifico (i pesi di una
  regressione lineare); altri lo trattano come una scatola chiusa e funzionano
  con qualunque cosa — li chiamiamo *agnostici*.

Tre domande, e ogni tecnica del capitolo trova il suo posto nella griglia.

`````

`````{tab} Superiore

Seguendo l'organizzazione di Molnar {cite}`molnar2022interpretable`,
distinguiamo lungo tre assi.

- **Intrinseca vs post-hoc.** L'interpretabilità *intrinseca* è una proprietà
  strutturale del modello, vincolato a priori a essere leggibile: regressione
  lineare/logistica, alberi poco profondi, sistemi a regole. L'interpretabilità
  *post-hoc* si applica dopo l'addestramento a un modello già dato, senza
  alterarne l'architettura (per esempio calcolando importanze delle feature o
  surrogati locali).
- **Globale vs locale.** Una spiegazione *globale* descrive il comportamento del
  modello sull'intero dominio — quali feature contano in media, che forma ha la
  dipendenza. Una spiegazione *locale* riguarda una singola predizione
  $f(x_0)$: perché *questo* input ha prodotto *questa* uscita. I due livelli
  richiedono metodi diversi; un modello con superficie decisionale complessa può
  essere globalmente incomprensibile ma localmente approssimabile.
- **Model-specific vs model-agnostic.** Un metodo *specifico* sfrutta la
  struttura interna di una classe di modelli (i coefficienti di un GLM, i
  gradienti di una rete). Un metodo *agnostico* accede solo alla funzione
  input→output $f$ e resta valido per qualunque modello, al costo di stimare il
  comportamento per campionamento anziché leggerlo dai parametri.

I tre assi sono largamente indipendenti: LIME, che vedremo, è post-hoc, locale e
agnostico; i coefficienti di una regressione lineare sono intrinseci, globali e
specifici.

`````

L'esempio più pulito di interpretabilità *intrinseca* è un albero di decisione
poco profondo: la sua logica si stampa per intero, senza strumenti esterni.

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text

# Un modello intrinsecamente interpretabile: un albero volutamente basso
iris = load_iris()
X, y = iris.data, iris.target
albero = DecisionTreeClassifier(max_depth=2, random_state=0)
albero.fit(X, y)

# Tutta la "logica" del modello è leggibile come una ricetta di if-then
print(export_text(albero, feature_names=list(iris.feature_names)))
```

Il testo stampato è una manciata di regole annidate sulla larghezza dei petali:
nessuna spiegazione post-hoc, il modello *è* la propria spiegazione. Se
lo stesso dato lo diamo in pasto a una foresta casuale da centinaia di alberi,
l'accuratezza sale ma la leggibilità sparisce — ed è lì che nascono i metodi del
capitolo.

## Spiegare non è essere fedeli

C'è una trappola, e il campo ci è caduto più di una volta. Una spiegazione
post-hoc è, per sua natura, una *ricostruzione a posteriori* di ciò che il
modello ha fatto — non il modello stesso. E una ricostruzione può essere
**plausibile senza essere fedele**: raccontarci una storia convincente sul
perché di una decisione, mentre il modello dentro faceva tutt'altro.

`````{tab} Elementare

Immagina di chiedere a un amico perché ha scelto un ristorante e lui ti
risponde: «per le recensioni». Suona sensato — ma se la verità è che ci andava
la sua ex e voleva rivederla, la spiegazione è *credibile* e *falsa* insieme.
Con i modelli è lo stesso: uno strumento post-hoc può produrre una motivazione
che a noi umani sembra ragionevole, ma che non corrisponde a come il modello ha
davvero deciso. Chiamiamo **fedeltà** quanto la spiegazione aderisce al vero
funzionamento del modello, e **plausibilità** quanto ci convince. Sono due cose
diverse, e la seconda è pericolosa proprio quando non c'è la prima: una
spiegazione bella e infedele ci fa fidare di un modello che non lo merita.

`````

`````{tab} Superiore

Una spiegazione post-hoc costruisce un modello surrogato $g$, interpretabile, che
approssima la scatola nera $f$ in un intorno del punto di interesse. La sua
qualità si misura con la **fedeltà locale**: il grado di accordo tra $g$ e $f$
sui punti $x'$ campionati vicino a $x_0$,

$$
\mathrm{fedelt\grave{a}}(g; x_0) = \mathbb{E}_{x' \sim \pi_{x_0}}
\big[\, \mathbb{1}\!\left(g(x') = f(x')\right) \,\big],
$$

dove $\pi_{x_0}$ è una distribuzione di prossimità centrata su $x_0$ e
$\mathbb{1}(\cdot)$ vale 1 quando surrogato e modello concordano. Una fedeltà
alta *sull'intorno* non garantisce nulla *globalmente*, ed è del tutto scorrelata
dalla **plausibilità** — quanto la spiegazione appare sensata a un umano.
Nulla vieta a un surrogato di essere plausibile e infedele, o fedele e
controintuitivo. È il difetto costitutivo di ogni spiegazione post-hoc:
approssima, e un'approssimazione può ingannare.

`````

Da qui uno dei dibattiti più netti del campo. Cynthia Rudin
{cite}`rudin2019stop` sostiene una tesi tagliente: per le **decisioni ad alto
rischio** — giustizia, sanità, credito — si dovrebbe **smettere di spiegare le
scatole nere** e usare invece **modelli intrinsecamente interpretabili**.
L'argomento è duplice. Primo: una spiegazione post-hoc infedele è peggio di
niente, perché dà una falsa sensazione di controllo. Secondo, e più radicale:
il presunto compromesso tra accuratezza e interpretabilità spesso *non esiste*
— su molti problemi con feature ben strutturate, un modello trasparente
raggiunge la stessa accuratezza di uno opaco, e allora l'opacità è un costo
senza contropartita.

Non tutti concordano. Su dati non strutturati — immagini, testo, segnali — le
reti profonde restano di gran lunga le più accurate, e rinunciarvi non è
un'opzione: lì il post-hoc è, pragmaticamente, l'unica finestra disponibile
sulla scatola nera. Il monito di Doshi-Velez e Kim {cite}`doshi2017towards`
vale per entrambe le fazioni: qualunque spiegazione va prodotta e *valutata*
con rigore scientifico, dichiarando cosa misura e con quale metodo, non offerta
come rassicurazione qualitativa. Non esistono spiegazioni «gratis»: esistono
spiegazioni verificate e spiegazioni che ci raccontiamo.

## Come è organizzato il capitolo

Con questa mappa in tasca, il capitolo procede lungo i tre assi.

Cominceremo dai **modelli trasparenti** — regressione lineare e logistica,
alberi di decisione, sistemi a regole — già incontrati nel capitolo sul Machine
Learning, riletti qui per la domanda «quanto sono leggibili?», e dalle misure di
**importanza delle feature** (globali) che dicono su cosa un modello si appoggia
in media. Passeremo poi alle **spiegazioni locali** — perché *questa* singola
predizione: **LIME**, il metodo del caso husky, che approssima il modello con un
surrogato lineare nell'intorno del punto; **SHAP**, che distribuisce il
«merito» di una predizione tra le feature con una regola presa dalla teoria dei
giochi; e le spiegazioni **controfattuali**, che rispondono a «cosa sarebbe
dovuto cambiare nell'input perché la decisione fosse diversa?». Chiuderemo con
l'**attribuzione nelle reti profonde** — saliency map, gradienti integrati,
mappe di attenzione che avevamo visto nascere nel capitolo sui Transformer — e
con la giovane **interpretabilità meccanicistica**, che prova a smontare una
rete circuito per circuito, come un ingegnere inverso apre un chip per capirne
la logica.

Un filo, sopra a tutto, tiene insieme il capitolo con quello sull'**AI
responsabile**: aprire la scatola nera non è un vezzo accademico, ma il primo
passo per costruire sistemi di cui potersi fidare — e da poter contestare quando
sbagliano. Il classificatore che guardava la neve non era un modello cattivo:
era un modello di cui nessuno aveva ancora guardato dentro.

```{admonition} Da ricordare
:class: important
- Un modello accurato può esserlo **per la ragione sbagliata** (effetto *Clever
  Hans*, la neve al posto del lupo): l'accuratezza aggregata non smaschera le
  **correlazioni spurie**, l'interpretabilità sì.
- Si spiega per **fiducia, debug, equità, scoperta scientifica e obblighi
  normativi**; la spiegazione «buona» dipende da **a chi** serve — sviluppatore,
  utente finale, regolatore vogliono cose diverse.
- Tre assi ordinano il campo: **intrinseca vs post-hoc**, **globale vs locale**,
  **model-specific vs model-agnostic**. Sono largamente indipendenti.
- **Plausibilità ≠ fedeltà**: una spiegazione post-hoc può convincere senza
  aderire a come il modello decide davvero. È il rischio strutturale
  dell'approssimazione a posteriori.
- Il dibattito: Rudin invita a **usare modelli interpretabili** per le decisioni
  ad alto rischio invece di spiegare scatole nere; sui dati non strutturati il
  post-hoc resta l'unica finestra. In ogni caso, spiegazioni **valutate con
  rigore** (Doshi-Velez & Kim), non rassicurazioni qualitative.
```
