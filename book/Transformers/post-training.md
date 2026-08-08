# Dopo il pre-addestramento: istruzioni, preferenze, allineamento

Prova a chiedere a un modello *solo* pre-addestrato: «Scrivi una poesia sul
mare». Una risposta perfettamente plausibile è: «— disse la maestra alla
classe, richiudendo il registro». Non è un guasto: è il compito che gli
abbiamo insegnato. Un modello pre-addestrato **completa** il testo nel modo
più probabile, e sul web quella frase compare più spesso dentro un racconto
scolastico che all'inizio di una poesia. GPT-3 {cite}`brown2020language` era
esattamente questo: un completatore geniale, capace di proseguire qualunque
testo, ma senza la minima nozione di cosa significhi *rispondere* a qualcuno.

Tra GPT-3 (2020) e ChatGPT (novembre 2022) il salto che tutti hanno percepito
non è (o non è solo) questione di scala. È il **post-training**: una seconda
fase di addestramento, molto più corta e mirata, che trasforma il completatore
in un assistente. La prova più eloquente sta nel paper di InstructGPT
{cite}`ouyang2022training`, il fratello maggiore di ChatGPT: nelle valutazioni
umane, le risposte di un modello da 1,3 miliardi di parametri rifinito con il
post-training venivano *preferite* a quelle del GPT-3 da 175 miliardi (un
modello più di cento volte più grande). Quel che manca al gigante non sono le
conoscenze: è la disposizione a usarle per aiutarti.

La ricetta, schematizzata in {numref}`fig-post-training-pipeline`, ha due
mosse principali: prima si insegna il *formato* con esempi svolti
(l'**instruction tuning**), poi si affina il *gusto* con i giudizi umani
(l'apprendimento dalle **preferenze**, via reinforcement learning o con la
scorciatoia DPO). Chiuderemo con un terzo ingrediente, più recente: spendere
più calcolo *al momento della risposta*, facendo «ragionare» il modello prima
di rispondere.

```{figure} ../figures/post-training-pipeline.svg
:name: fig-post-training-pipeline
:alt: "Pipeline a quattro stadi del post-training: pre-addestramento, SFT su coppie istruzione-risposta, preferenze umane con reward model e PPO sotto vincolo KL, modello assistente finale; una freccia tratteggiata indica la DPO come scorciatoia che salta il reward model."
:width: 100%

Dal completatore all'assistente: SFT sugli esempi svolti, poi preferenze
umane (reward model + RL, oppure la scorciatoia DPO che salta il giudice).
```

## Studiare gli esempi svolti: l'instruction tuning

```{figure} ../figures/instruction-tuning.svg
:name: fig-instruction-tuning
:alt: "Lo stesso prompt dato a due modelli. Il modello base lo prosegue come farebbe un testo trovato sul web, generando altre domande simili invece di rispondere. Il modello dopo instruction tuning lo interpreta come una consegna ed esegue, producendo la risposta richiesta."
:width: 96%

Stesso prompt, due comportamenti. Il modello base non è più ignorante: sta
facendo esattamente ciò per cui era stato addestrato, cioè proseguire il
testo. L'instruction tuning gli insegna che quel testo era un ordine.
```

La differenza mostrata in {numref}`fig-instruction-tuning` è la ragione per
cui il post-training esiste: fra un modello che completa e un assistente che
esegue non c'è più conoscenza, c'è una diversa interpretazione della richiesta.

Il primo passo si chiama **SFT** (*supervised fine-tuning*), o *instruction
tuning*: si raccoglie un dataset di coppie (istruzione, risposta) scritte da
persone, «Riassumi questo articolo» seguito da un buon riassunto, «Traduci in
inglese: il gatto nero salta sul muro» seguito da *«The black cat jumps on the
wall»*, e si continua l'addestramento del modello su questi esempi, con la
stessa identica tecnica del pre-addestramento.

`````{tab} Elementare

Immagina un apprendista che ha passato dieci anni a leggere *tutta* la
biblioteca del suo mestiere: manuali, riviste, verbali, romanzi. Sa
moltissimo, ma nessuno gli ha mai mostrato com'è fatto il lavoro vero e
proprio: se gli chiedi qualcosa, ti recita il seguito più probabile della tua
frase, come un'eco istruita. L'instruction tuning è il tirocinio: gli mettiamo
davanti qualche migliaio di **compiti già svolti bene** (la domanda di un
cliente con accanto la risposta di un professionista esperto) e lui li studia
uno per uno. Non impara quasi nulla di nuovo sul mondo: quello l'aveva già
letto in biblioteca. Impara il **formato**: che quando arriva un'istruzione,
la cosa da fare non è continuarla, ma eseguirla. È un tirocinio
sorprendentemente breve (migliaia di esempi contro i miliardi di frasi della
biblioteca) proprio perché non aggiunge sapere: orienta quello che c'è già.

`````

`````{tab} Superiore

Sia $\mathcal{D} = \{(x^{(i)}, y^{(i)})\}$ un dataset di coppie
istruzione–risposta. La SFT minimizza la stessa cross-entropia autoregressiva
del pre-addestramento, ma applicata ai soli token della risposta:

$$
\mathcal{L}_{\text{SFT}}(\theta) =
-\sum_{(x,y)\in\mathcal{D}} \sum_{t=1}^{|y|}
\log \pi_\theta\big(y_t \mid x,\, y_{<t}\big),
$$

dove $\pi_\theta$ è il modello di linguaggio con parametri $\theta$, $x$ è
l'istruzione (il *prompt*), $y_t$ è il $t$-esimo token della risposta e
$y_{<t}$ sono i token che lo precedono. In pratica i token del prompt vengono
*mascherati* nella loss: il modello li legge ma non viene penalizzato su di
essi, perché non vogliamo insegnargli a generare domande, bensì risposte.
L'ordine di grandezza dei dati è minuscolo rispetto al pre-addestramento: per
InstructGPT bastarono circa 13 000 dimostrazioni scritte da annotatori
{cite}`ouyang2022training`. Il limite strutturale è quello di ogni *behaviour
cloning*: il modello impara a imitare le dimostrazioni, non a distinguere una
risposta eccellente da una mediocre, e per molte richieste («scrivi una poesia
sul mare») non esiste *la* risposta giusta da fargli copiare.

`````

Proprio qui la SFT si ferma. Per andare oltre serve un'osservazione quasi
banale: per un essere umano **giudicare è più facile che scrivere**. Pochi di
noi saprebbero comporre una bella poesia sul mare; quasi tutti, davanti a due
poesie, sanno dire quale preferiscono. Il post-training moderno è costruito
su questa asimmetria.

## Adattare senza riaddestrare tutto: LoRA

Prima di proseguire, un problema pratico. Tutto quello che abbiamo descritto
(e tutto ciò che segue) presuppone di poter aggiornare i pesi del modello. Ma
un modello da $7$ miliardi di parametri a $32$ bit occupa circa $28$ GB solo
per i pesi (quattro byte a parametro) e l'addestramento ne richiede il triplo
abbondante fra gradienti e stati dell'ottimizzatore. Fuori dai laboratori,
quasi nessuno può permetterselo.

```{figure} ../figures/lora-fine-tuning-efficiente.svg
:name: fig-lora
:alt: "Schema di LoRA: la matrice dei pesi pre-addestrati W resta congelata e riceve l'ingresso; accanto a essa due matrici piccole e addestrabili, A e B, formano un percorso parallelo a basso rango. Le uscite dei due rami si sommano prima di proseguire. Solo A e B ricevono gradiente."
:width: 78%

LoRA non tocca $W$: gli affianca una scorciatoia stretta. Il ramo parallelo ha
pochi parametri perché passa da un collo di bottiglia, e solo quelli si
addestrano.
```

La forma di {numref}`fig-lora` spiega anche perché l'adattamento si possa
*staccare*. Se ciò che si è imparato vive tutto in $A$ e $B$, e $W$ è rimasta
identica, allora un adattamento è un file piccolo che si aggiunge o si toglie:
lo stesso modello base può servire compiti diversi cambiando solo il ramo
laterale.

`````{tab} Elementare

**LoRA** (*Low-Rank Adaptation*, Hu e colleghi, 2021) parte da un'osservazione:
quando adatti un modello già addestrato a un compito nuovo, i pesi non cambiano
in modo disordinato. Si spostano poco, e in modo molto **strutturato**.

L'idea è quindi congelare il modello originale (non si tocca) e affiancargli
una piccola correzione addestrabile. Invece di riscrivere una matrice di pesi
enorme, se ne impara una versione «compressa» fatta di due matrici sottili, il
cui prodotto ha la stessa forma dell'originale ma molti meno numeri da
imparare.

L'analogia è il lucido da architetto: la pianta originale resta intatta, tu
disegni le modifiche su un foglio trasparente sovrapposto. Puoi tenere molti
lucidi diversi (uno per il supporto clienti, uno per il codice, uno per il
tono formale) e cambiarli in un istante sullo stesso disegno di base.

In pratica si addestra spesso **meno dello 0,1%** dei parametri, il file da
salvare pesa megabyte invece di gigabyte, e la qualità resta vicina al
fine-tuning completo.

`````

`````{tab} Superiore

Data una matrice di pesi pre-addestrata $W_0 \in \mathbb{R}^{d\times k}$, LoRA
non la modifica: parametrizza l'aggiornamento come prodotto di due matrici a
rango basso,

$$
W = W_0 + \Delta W = W_0 + \frac{\alpha}{r}\,B A,
\qquad B \in \mathbb{R}^{d\times r},\ A \in \mathbb{R}^{r\times k},\ r \ll \min(d,k).
$$

Solo $A$ e $B$ ricevono gradiente. I parametri addestrabili passano da $dk$ a
$r(d+k)$: per $d=k=4096$ e $r=8$ si scende da $16{,}8$ milioni a $65\,536$ per
matrice, lo $0{,}39\%$. All'inizio $A$ è inizializzata casualmente e $B$ a zero,
così $\Delta W = 0$ e il modello parte esattamente dal comportamento
pre-addestrato; $\alpha/r$ è un fattore di scala che disaccoppia il *learning
rate* efficace dalla scelta di $r$.

Tre conseguenze pratiche:

1. **Nessuna latenza aggiuntiva in inferenza.** A differenza degli adapter
   inseriti in serie, $BA$ si può sommare a $W_0$ una volta per tutte prima del
   deployment: il grafo di calcolo torna identico all'originale.
2. **Adattatori componibili e leggeri.** Si tengono in memoria molti LoRA
   sullo stesso modello di base e si scambiano per richiesta: è il meccanismo
   dietro il *multi-tenant serving* di modelli specializzati.
3. **QLoRA** (Dettmers e colleghi, 2023) porta l'idea all'estremo: il modello
   base viene quantizzato a $4$ bit e congelato, gli adattatori restano in
   precisione più alta. Consente il fine-tuning di modelli da decine di miliardi
   di parametri su una sola GPU consumer.

Il limite è dove ci si aspetta: LoRA **adatta**, non insegna. Per far
acquisire al modello conoscenza sostanzialmente nuova, o per cambiarne il
comportamento in profondità, il rango basso è un collo di bottiglia, e lì
serve il fine-tuning completo.

`````

## Il giudizio umano come segnale: RLHF

L'idea non nasce con i modelli di linguaggio. Nel 2017 Christiano e colleghi
{cite}`christiano2017deep` insegnano a un robottino simulato a fare il salto
mortale all'indietro (un comportamento per cui nessuno sa scrivere una
funzione di ricompensa a mano) mostrando a un valutatore umano coppie di brevi
video e chiedendogli solo: *quale dei due somiglia di più a un salto mortale?*
Bastarono circa 900 confronti, meno di un'ora di tempo umano.

```{figure} ../figures/deep-rl-human-preferences-2017.svg
:name: fig-preferenze-umane
:alt: "Ciclo chiuso in quattro stazioni: l'agente di reinforcement learning genera coppie di traiettorie; una persona guarda le due e sceglie la preferita; da queste scelte un modello di ricompensa impara a dare punteggi; il modello di ricompensa restituisce all'agente una ricompensa predetta, che lo riaddestra, e il giro ricomincia."
:width: 90%

Il giro che sostituisce la funzione di ricompensa scritta a mano. La persona
non spiega mai cosa sia un salto mortale: si limita a preferire, e il modello
di ricompensa deduce il resto.
```

Il passaggio decisivo di {numref}`fig-preferenze-umane` è il modello di
ricompensa in mezzo. Senza di lui ogni passo di addestramento richiederebbe
un giudizio umano, il che è impraticabile; con lui i confronti servono a
insegnare *una volta* un giudice artificiale, che poi lavora quanto serve. La tecnica si
chiama **RLHF** (*Reinforcement Learning from Human Feedback*), e con
InstructGPT {cite}`ouyang2022training` viene applicata in grande al
linguaggio, in due tempi: prima i confronti umani addestrano un **reward
model**, un modello che impara a dare voti; poi il reward model fa da giudice
automatico mentre il modello di linguaggio viene ottimizzato con il
reinforcement learning.

`````{tab} Elementare

Pensa a un ristorante che vuole perfezionare un piatto. Assumere un critico
che *descriva a parole* il piatto perfetto è impossibile; far assaggiare due
versioni e chiedere «quale preferisci?» è facilissimo. Si procede così:
l'assaggiatore confronta centinaia di coppie di piatti, e da tutti quei
confronti si distilla una specie di **palato artificiale** (un giudice
automatico che, assaggiato un piatto qualsiasi, gli dà un voto coerente con i
gusti raccolti). A quel punto il cuoco può lavorare anche di notte, senza
l'assaggiatore: prova una variante, il palato artificiale la vota, e lui
aggiusta la ricetta per far salire il voto. Con una regola d'oro appesa in
cucina: **mai stravolgere la ricetta di partenza**. Perché il palato
artificiale è un'imitazione, e ha punti ciechi: se il cuoco insegue solo il
voto, prima o poi scopre che (che so), raddoppiare la panna inganna il
giudice, e finisce per servire piatti assurdi che «prendono voti alti» ma che
nessun cliente vero vorrebbe. La regola del «resta vicino alla ricetta» tiene
la creatività al guinzaglio: piccoli aggiustamenti sì, stravolgimenti no.

`````

`````{tab} Superiore

**Fase 1: il reward model.** Per un prompt $x$ si generano due risposte e un
annotatore indica la preferita, $y_w$ (*winner*), contro la scartata, $y_l$
(*loser*). Il reward model $r_\phi(x, y)$ (tipicamente lo stesso Transformer
con una testa scalare al posto della softmax) viene addestrato assumendo il
modello di **Bradley–Terry** (1952), per cui la probabilità di preferenza
dipende dalla differenza dei punteggi:

$$
P(y_w \succ y_l \mid x) = \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big),
$$

dove $\sigma$ è la sigmoide e $\phi$ sono i parametri del reward model. Se ad
esempio la differenza di punteggio è $1{,}1$, il modello assegna alla
preferenza osservata probabilità $\sigma(1{,}1) \approx 0{,}75$. La loss è la
log-verosimiglianza negativa dei confronti raccolti.

**Fase 2: la policy.** Il modello di linguaggio diventa una *policy*
$\pi_\theta$ nel senso del reinforcement learning (il prompt è lo stato, la
risposta generata è l'azione) e si ottimizza

$$
\max_\theta\;
\mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta}\big[ r_\phi(x, y) \big]
\;-\; \beta\,
D_{\mathrm{KL}}\big(\pi_\theta(\cdot \mid x) \,\|\, \pi_{\text{ref}}(\cdot \mid x)\big),
$$

dove $\pi_{\text{ref}}$ è il modello di riferimento congelato (di solito il
modello SFT), $D_{\mathrm{KL}}$ è la divergenza di Kullback–Leibler
{cite}`kullback1951information` vista nel capitolo sui richiami di matematica
e $\beta > 0$ regola la forza del vincolo. La penalità KL serve a due cose:
impedisce alla policy di derivare verso le zone in cui $r_\phi$ (addestrato su
dati limitati) estrapola male (il *reward hacking* su cui torneremo), e
preserva la fluidità linguistica accumulata nel pre-addestramento.
L'ottimizzazione usa **PPO** {cite}`schulman2017proximal`, l'algoritmo a
gradiente di policy che hai visto sviluppato, insieme a tutta la famiglia dei
*policy gradient*, nel capitolo sul Deep Reinforcement Learning: l'idea in una
riga è aumentare la probabilità delle risposte con ricompensa alta, a piccoli
passi controllati per non destabilizzare la policy.

`````

Vale la pena fissare il punto d'incontro: aumentare la probabilità delle
azioni che ricevono un giudizio positivo è la stessa meccanica che, come
vedrai nel capitolo sul Deep Reinforcement Learning, fa vincere partite di
Go; qui insegna a un modello di linguaggio a essere utile. La «mossa» è
un'intera risposta, e il punteggio non viene dalle regole di un gioco ma da
un modello addestrato a imitare i gusti di valutatori in carne e ossa.

## DPO: imparare dalle preferenze senza il giudice

L'RLHF funziona, ma è un cantiere pesante: bisogna tenere in memoria quattro
reti (la policy, il riferimento congelato, il reward model e il critico di
PPO) e addestrarne tre, e il reinforcement learning su testo è notoriamente
capriccioso da stabilizzare. Nel 2023 Rafailov e colleghi
{cite}`rafailov2023direct` mostrano che si può arrivare quasi allo stesso
punto con una semplice loss supervisionata. Il titolo del paper è già la tesi:
*Your Language Model is Secretly a Reward Model*; il tuo modello di linguaggio
è, a sua insaputa, già un reward model. Il metodo si chiama **DPO** (*Direct
Preference Optimization*).

`````{tab} Elementare

Torniamo in cucina. Il metodo classico prevedeva due tempi: prima addestrare
un giudice artificiale sui confronti degli assaggiatori, poi far cucinare il
cuoco per il giudice. La DPO si accorge che il giro è più lungo del
necessario: il cuoco può **saltare il giudice** e imparare direttamente dai
confronti. Per ogni coppia già valutata (piatto preferito, piatto scartato),
ritocca la ricetta in modo da rendere un po' più probabile il preferito e un
po' meno probabile lo scartato. E il ritocco è dosato con intelligenza: se il
cuoco *già* favorisce il piatto giusto, il confronto non insegna quasi nulla e
la correzione è minima; se invece sta ancora dalla parte sbagliata, la
correzione è energica. Anche la regola d'oro sopravvive, incorporata nel
metodo: i ritocchi si misurano sempre *rispetto alla ricetta di partenza*,
così il cuoco migliora senza stravolgere. Stessa destinazione dell'RLHF, senza
l'intermediario, e senza il cantiere.

`````

`````{tab} Superiore

Il punto di partenza è un fatto notevole: l'obiettivo RLHF con penalità KL,
se lo si massimizza fra *tutte* le policy possibili e non solo dentro la
classe parametrica di $\pi_\theta$, ha una soluzione ottima in forma chiusa,

$$
\pi^*(y \mid x) = \frac{1}{Z(x)}\,
\pi_{\text{ref}}(y \mid x)\,
\exp\!\Big(\tfrac{1}{\beta}\, r(x, y)\Big),
$$

dove $Z(x)$ è la costante di normalizzazione. Invertendo la relazione, la
ricompensa si può scrivere in funzione della policy ottima:
$r(x,y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} +
\beta \log Z(x)$. Sostituendo questa espressione nel modello di
Bradley–Terry, la costante $Z(x)$ si cancella nella differenza tra le due
risposte, e la verosimiglianza delle preferenze diventa una loss che dipende
*solo dalla policy*:

$$
\mathcal{L}_{\text{DPO}}(\theta) =
-\,\mathbb{E}_{(x,\, y_w,\, y_l) \sim \mathcal{D}}
\left[
\log \sigma\!\left(
\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)}
\;-\;
\beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}
\right)
\right],
$$

dove $\mathcal{D}$ è il dataset di terne (prompt, risposta preferita,
risposta scartata); $x$ è il prompt, $y_w$ la risposta preferita e $y_l$
quella scartata; $\pi_\theta$ è la policy in addestramento (l'unica di cui si
aggiornano i parametri $\theta$); $\pi_{\text{ref}}$ è il riferimento
congelato, di norma il modello SFT; $\beta > 0$ (valori tipici tra $0{,}1$ e
$0{,}5$) controlla la forza del vincolo implicito verso il riferimento, come
la penalità KL dell'RLHF; $\sigma$ è la sigmoide. La quantità
$\hat{r}_\theta(x,y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$
è la **ricompensa implicita**: la loss è una regressione logistica che chiede
alla ricompensa implicita della risposta preferita di superare quella della
scartata. Il gradiente pesa ogni coppia per quanto il modello la sbaglia
ancora: i confronti già «vinti» contribuiscono poco, quelli persi molto.
Niente reward model esplicito, niente campionamento, niente PPO: un normale
addestramento supervisionato su coppie. L'equivalenza con l'RLHF è esatta
solo in quel limite non parametrico, e sulla distribuzione delle coppie
raccolte: con una policy parametrica e coppie fissate una volta per tutte (la
DPO non campiona mai da $\pi_\theta$, il PPO sì) i due metodi in pratica
divergono, ed è qui che va cercata la differenza fra i loro risultati.

`````

La loss DPO è così compatta che possiamo scriverla per intero. La funzione
riceve le log-probabilità totali delle risposte (la somma dei logaritmi delle
probabilità dei loro token) sotto la policy e sotto il riferimento:

```python
import torch
import torch.nn.functional as F

def dpo_loss(logp_w_policy, logp_l_policy,
             logp_w_ref, logp_l_ref, beta=0.1):
    """Loss DPO su un batch di coppie (preferita, scartata).

    Ogni argomento e' la log-probabilita' totale della risposta:
    somma dei log-prob dei suoi token, ottenuta con log_softmax
    sui logits del modello. Il riferimento e' congelato (no grad).
    """
    # ricompensa implicita: quanto ciascun modello "favorisce" la risposta
    margine_w = logp_w_policy - logp_w_ref   # risposta preferita
    margine_l = logp_l_policy - logp_l_ref   # risposta scartata
    # la preferita deve staccare la scartata: regressione logistica
    return -F.logsigmoid(beta * (margine_w - margine_l)).mean()

# tensori fittizi: log-prob totali di 4 coppie di risposte
logp_w_policy = torch.tensor([-12.3, -45.1,  -8.7, -30.2])
logp_l_policy = torch.tensor([-11.9, -47.8,  -9.5, -29.8])
logp_w_ref    = torch.tensor([-12.5, -46.0,  -8.9, -30.5])
logp_l_ref    = torch.tensor([-11.7, -46.5,  -9.1, -30.1])

print(dpo_loss(logp_w_policy, logp_l_policy, logp_w_ref, logp_l_ref))
# tensor(0.6548): poco sotto log(2) ~ 0.693, apprendimento appena iniziato
```

I numeri fittizi nascondono un dettaglio istruttivo. Nella prima coppia la
policy, in assoluto, assegna log-prob più alta alla risposta *scartata*
($-11{,}9$ contro $-12{,}3$), ma alla DPO non importa: conta il confronto col
riferimento, e rispetto a $\pi_{\text{ref}}$ la preferita ha guadagnato
terreno (margine $+0{,}2$), mentre la scartata ne ha perso (margine $-0{,}2$):
un divario di $0{,}4$ a favore della preferita. Nella quarta coppia i due
margini si equivalgono: è la coppia «non ancora imparata», quella su cui il
gradiente spinge di più. E la SFT? Non merita codice nuovo: è il training loop
del capitolo su PyTorch, con la cross-entropia calcolata sui token della
risposta (lo stesso ciclo `forward`, `loss`, `backward`, `step` che ormai
conosci a memoria).

## Pensare prima di rispondere: il calcolo al momento dell'inferenza

C'è un terzo asse, ortogonale ai primi due: invece di (o oltre a) migliorare
i *pesi*, si può spendere più *calcolo al momento della risposta*.

```{figure} ../figures/reasoning-test-time-compute.svg
:name: fig-test-time-compute
:alt: "Grafico con il tempo di riflessione concesso al modello in ascissa e l'accuratezza in ordinata. La curva di un modello che risponde subito resta piatta: concedergli più tempo non cambia nulla. La curva di un modello addestrato a ragionare sale invece al crescere del tempo, continuando a migliorare ben oltre il punto in cui l'altra si è fermata."
:width: 92%

Un secondo asse su cui spendere. La curva piatta è il punto: dare più tempo
non basta, il modello deve essere stato addestrato a usarlo.
```

Le due curve di {numref}`fig-test-time-compute` distinguono due cose che si
confondono facilmente. Non è che «pensare di più» aiuti sempre: aiuta se il
modello ha imparato a spendere quei token in passaggi che si costruiscono
l'uno sull'altro. Altrimenti il tempo in più produce solo testo in più. La chiave
di volta è la **chain-of-thought** («catena di pensiero»), documentata da Wei
e colleghi nel 2022 {cite}`wei2022chain`: per i problemi che richiedono più
passaggi, far generare al modello il ragionamento intermedio prima della
risposta migliora nettamente l'accuratezza.

`````{tab} Elementare

È la regola che conosci dal compito di matematica: «mostra i passaggi». Alla
domanda «un treno parte alle 9:47 e arriva alle 11:23: quanto dura il
viaggio?», sparare il risultato a colpo d'occhio fa sbagliare spesso; scrivere
i passaggi, da 9:47 a 10:00 sono 13 minuti, poi un'ora fino alle 11:00, poi
altri 23: totale 96 minuti, cioè 1 ora e 36 (porta quasi sempre al risultato
giusto). Con i modelli funziona allo stesso modo: se l'esempio che gli mostri
contiene i passaggi, o se glieli chiedi esplicitamente, il modello li scrive e
sbaglia meno, perché ogni passaggio può appoggiarsi ai precedenti invece di
indovinare tutto in un colpo. Un raffinamento semplice: fargli risolvere lo
stesso problema più volte per strade diverse e prendere la risposta più
votata, come chiedere a tre amici e fidarsi della maggioranza. I modelli
«ragionanti» usciti tra il 2024 e il 2025 portano l'idea alle conseguenze:
sono addestrati a produrre da soli, prima di ogni risposta, una lunga brutta
copia di passaggi, che costa tempo e calcolo in più, ripagati soprattutto in
matematica e programmazione, dove la risposta si può verificare.

`````

`````{tab} Superiore

Nel *chain-of-thought prompting* gli esempi nel prompt includono i passaggi
intermedi, e il modello li riproduce prima della risposta finale. È una
capacità **emergente con la scala**: sotto una certa dimensione le catene non
aiutano o peggiorano, mentre con PaLM da 540 miliardi di parametri otto esempi
con catena bastarono a superare, sul benchmark di problemi aritmetici GSM8K,
persino un GPT-3 rifinito ad hoc con verificatore {cite}`wei2022chain`. La
*self-consistency* aggiunge un passo: si campionano più catene indipendenti e
si sceglie la risposta finale a maggioranza (Wang et al., 2022). I modelli
«ragionanti», o1 di OpenAI (settembre 2024), DeepSeek-R1
{cite}`guo2025deepseek` a pesi aperti (gennaio 2025), interiorizzano la
catena: vengono addestrati con reinforcement learning su problemi a **risposta
verificabile** (correttezza del risultato matematico, superamento dei test per
il codice), dove la ricompensa non richiede giudizi umani. DeepSeek-R1-Zero
mostra che il solo RL, senza SFT preliminare, fa emergere comportamenti di
auto-verifica e ripensamento dei propri passaggi. Il quadro consolidato, senza
estrapolazioni: i guadagni sono concentrati nei domini verificabili; il costo
per risposta cresce con la lunghezza della catena (più token, più latenza); e
su quanto queste catene corrispondano a un «ragionamento» in senso proprio il
dibattito scientifico resta aperto; prudenza nell'attribuirvi troppo è buona
epistemologia, oltre che buon gusto.

`````

## Quel che il giudice non vede

Chiudiamo con l'onestà che questo libro deve al lettore: il post-training
migliora i modelli, ma non è una soluzione, e i suoi difetti hanno nomi
precisi.

Il primo è il **reward hacking**: quando ottimizzi un surrogato del tuo vero
obiettivo, prima o poi ottieni il surrogato e perdi l'obiettivo. Il reward
model imita i giudizi umani, e i giudizi umani hanno debolezze sistematiche:
tendiamo a premiare le risposte lunghe, sicure di sé, ben impaginate. Un
modello ottimizzato contro quel giudice impara la prolissità e la sicurezza
esibita *prima ancora* dell'utilità: massimizza il voto, non il valore. La
penalità KL mitiga, non guarisce.

Il secondo è la **ruffianeria** (*sycophancy*), documentata empiricamente
(Sharma et al., 2023): se i valutatori preferiscono (anche solo un po' più
spesso) le risposte che danno loro ragione, il modello impara a dare ragione.
Contraddici un assistente addestrato sulle preferenze e spesso ritratterà una
risposta corretta, perché nei dati di confronto l'accordo vinceva sul
disaccordo. È l'esempio perfetto di ottimizzazione riuscita dell'obiettivo
sbagliato.

E c'è la domanda che nessuna loss può chiudere: **allineato a chi?** Le
«preferenze umane» dell'RLHF sono, in concreto, le preferenze di qualche
decina di annotatori che seguono le linee guida di un'azienda. Persone
diverse, culture diverse, contesti diversi preferiscono risposte diverse: la
scelta di quali giudizi contino è una decisione di chi costruisce il modello,
non un fatto tecnico. Per questo l'**allineamento** è oggi un'area di ricerca
a pieno titolo, non un ritocco finale: abbiamo strumenti per orientare il
comportamento dei modelli, non garanzie sul risultato.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Un modello appena pre-addestrato **completa** il testo, non risponde: è
  un'eco istruita. Il salto verso l'assistente è una seconda fase di
  addestramento, molto più corta e mirata. Quanto pesi lo dice un dato: nel
  giudizio delle persone un modello piccolo ma rifinito così batteva uno più
  di cento volte più grande {cite}`ouyang2022training`.
- **Il tirocinio**: qualche migliaio di compiti già svolti bene (una richiesta
  con accanto la risposta di un professionista), studiati uno per uno. Non
  aggiunge sapere, quello era già in biblioteca: insegna che a un'istruzione
  non si dà un seguito, si dà esecuzione.
- **Il palato artificiale** {cite}`christiano2017deep`: giudicare è più facile
  che scrivere, quindi alle persone si chiede solo quale di due risposte
  preferiscono; da quei confronti si distilla un giudice automatico, e il
  modello poi lavora per far salire il voto. Con una regola d'oro appesa in
  cucina: restare vicini alla ricetta di partenza, perché il giudice è
  un'imitazione e ha i suoi punti ciechi.
- **Saltare il giudice** {cite}`rafailov2023direct`: dagli stessi confronti si
  può imparare direttamente, rendendo un po' più probabile la risposta
  preferita e un po' meno quella scartata, e misurando sempre i ritocchi
  rispetto alla ricetta di partenza. Stessa destinazione, senza il cantiere.
- **Mostrare i passaggi** {cite}`wei2022chain`: scrivere il ragionamento prima
  della risposta fa sbagliare meno; rifare lo stesso problema per strade
  diverse e tenere la risposta più votata aiuta ancora; i modelli
  «ragionanti» {cite}`guo2025deepseek` si addestrano a stendere da soli una
  lunga brutta copia. Costa tempo e calcolo, e ripaga soprattutto dove la
  risposta si può verificare.
- Limiti aperti: il modello impara a **prendere voti alti** più che a essere
  utile (risposte lunghe, sicure di sé, ben impaginate), impara a **dare
  ragione** a chi lo contraddice, e resta la domanda che nessun addestramento
  chiude: allineato ai gusti di chi?
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Un modello pre-addestrato **completa**, non risponde: il salto verso
  l'assistente è il **post-training**. In InstructGPT
  {cite}`ouyang2022training` un modello da 1,3 miliardi di parametri
  allineato batteva, nel giudizio umano, il GPT-3 da 175 miliardi.
- **SFT / instruction tuning**: la stessa cross-entropia del
  pre-addestramento su coppie (istruzione, risposta) scritte da persone;
  insegna il *formato*, non nuove conoscenze.
- **RLHF** {cite}`christiano2017deep`: confronti umani → reward model
  (Bradley–Terry) → ottimizzazione con PPO e **penalità KL** verso il
  modello di partenza, per non finire nei punti ciechi del giudice.
- **DPO** {cite}`rafailov2023direct`: stessa sostanza senza RL esplicito; una
  loss supervisionata sulle coppie preferita/scartata, con la ricompensa
  implicita $\beta \log (\pi_\theta / \pi_{\text{ref}})$.
- **Test-time compute**: chain-of-thought {cite}`wei2022chain`,
  self-consistency, e i modelli «ragionanti» addestrati con RL su risposte
  verificabili {cite}`guo2025deepseek` (guadagni reali ma concentrati nei
  domini verificabili, a costo di più calcolo per risposta).
- Limiti aperti: **reward hacking**, **ruffianeria**, e la domanda non
  tecnica «allineato a chi?».
```
`````
