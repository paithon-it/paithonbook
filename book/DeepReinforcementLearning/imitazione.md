# Imparare guardando: imitazione e clonazione comportamentale

Nel 1989, alla Carnegie Mellon, un furgone attrezzato percorreva le strade
attorno al campus guidato da una rete neurale piccolissima, un solo strato di
neuroni fra l'ingresso e l'uscita. Si chiamava **ALVINN**, riceveva l'immagine
di una telecamera e la lettura di un telemetro laser (uno strumento che misura
a che distanza sono le cose), e restituiva l'angolo di sterzata
{cite}`pomerleau1989alvinn`. Non aveva imparato per tentativi ed errori, il che
sarebbe stato imprudente su una strada vera: era stato addestrato a
**riprodurre, per ogni immagine di strada, l'angolo di sterzata corretto**. Gli
si mostrava la risposta giusta e lo si correggeva finché non la indovinava, che
è il modo di imparare più comune di tutti e si chiama apprendimento
supervisionato.

C'è un dettaglio di quella storia che sembra un'inezia ed è invece tutto il
problema, arrivato con trent'anni di anticipo. Le immagini su cui ALVINN
imparò, nel lavoro del 1989, erano **simulate**. Quando Pomerleau passò alle
registrazioni di un guidatore vero, due anni dopo, si trovò davanti a un
ostacolo che dovette aggirare a mano: un guidatore bravo non esce mai dalla
corsia, quindi nelle sue registrazioni non c'è un solo fotogramma che mostri
come si rimedia a un'auto storta. Se li dovette fabbricare, deformando le foto
buone per ottenerne altre scattate come se l'auto fosse un po’ fuori centro
{cite}`pomerleau1991efficient`. Il resto della sezione spiega perché quel
lavoro in più non fosse un capriccio, ma la sola cosa che tenesse in piedi il
furgone.

L'imitazione è, in un certo senso, l'idea più ovvia di tutte, e per questo
conviene capire bene perché non basta. Nei capitoli precedenti l'agente impara
da una ricompensa, e la ricompensa è la parte difficile: scriverla per una
guida sicura o per una risposta utile è un problema aperto. Se però qualcuno
sa già fare il compito, quel problema si può aggirare. Non gli si chiede di
scrivere una funzione di ricompensa: gli si chiede di **fare vedere**.

## Clonare un comportamento è apprendimento supervisionato

`````{tab} Elementare

Si registra un esperto mentre lavora e si annota, momento per momento, che cosa
vedeva e che cosa ha fatto. Ne esce un elenco di coppie: «in questa situazione,
questa mossa». A quel punto il problema di reinforcement learning è sparito, e
al suo posto c'è un normalissimo problema di apprendimento supervisionato:
l'ingresso è la situazione, l'uscita da indovinare è la mossa dell'esperto.

Chi impara viene messo davanti a una situazione dell'elenco e propone la sua
mossa: si guarda di quanto ha mancato quella dell'esperto e lo si corregge un
poco, così che la volta dopo sia più vicino. Poi la situazione seguente, e
tutte le altre, molte volte di fila, finché lo scarto medio non è quasi zero.
Che la mossa sia una scelta fra poche (frenare, sterzare, tirare dritto) o un
numero da regolare (di quanti gradi girare il volante) cambia solo il modo di
misurare lo scarto, non l'impianto.

Si chiama **clonazione comportamentale**, ed è tanto semplice che l'abbiamo già
incontrata due volte senza chiamarla così. Il primo AlphaGo, prima di giocare
contro sé stesso, aveva imparato a proporre mosse guardando
centosessantamila partite di giocatori forti. E la prima fase
dell'addestramento di un assistente
conversazionale è esattamente questa: si raccolgono risposte scritte da persone
e si insegna al modello a scriverne di simili.

I pregi sono seri. È stabile, perché è addestramento
supervisionato ordinario: nessuno dei tormenti visti fin qui nei metodi che
imparano dei voti, cioè il bersaglio che si sposta mentre lo insegui e i voti
che crescono senza fermarsi della triade fatale.
È efficiente, perché ogni dimostrazione insegna qualcosa subito, mentre un
agente che esplora butta via migliaia di tentativi. Ed è sicura, perché non
serve far provare al sistema mosse a caso nel mondo vero.

C'è però una condizione che nessuno scrive, perché sembra ovvia: che le
situazioni da affrontare domani siano quelle dell'elenco di oggi. A sceglierle
non è il mondo. Dove ci si trova un istante dopo dipende dalla mossa appena
fatta, e le mosse di chi ha imparato non sono quelle dell'esperto.

`````

`````{tab} Superiore

Dato un insieme di dimostrazioni $\mathcal{D} = \{(s_i, a_i)\}$ prodotte da una
politica esperta $\pi^\star$, la **clonazione comportamentale** stima

$$
\hat\pi = \arg\min_{\pi} \; \mathbb{E}_{(s,a)\sim\mathcal{D}}
\big[\, \ell\big(\pi(s),\, a\big) \,\big],
$$

con $\ell$ la cross-entropia per azioni discrete o l'errore quadratico per
azioni continue. Non compaiono ricompense, non compare l'equazione di Bellman,
non compare il bootstrapping: è **regressione o classificazione**, con tutto
ciò che ne consegue in termini di stabilità e di strumenti già noti.

La stessa mossa compare in tre punti del libro, e conviene riconoscerla. Nella
{doc}`sezione sui metodi a gradiente di policy <policy-gradient>`, la rete di
policy di AlphaGo è
pre-addestrata in modo supervisionato su partite umane prima del *self-play*.
Nel post-addestramento dei modelli linguistici, la fase di *supervised
fine-tuning* che precede l'RLHF è clonazione comportamentale su dimostrazioni
scritte da persone. E nella {doc}`sezione sull'RL offline <offline-rl>`, la
componente supervisionata del Decision Transformer è la stessa cosa,
condizionata sul ritorno desiderato.

L'assunzione nascosta, che è tutto il problema, è quella di ogni apprendimento
supervisionato: **dati indipendenti e identicamente distribuiti**. Qui non lo
sono, perché gli stati che l'agente incontrerà dipendono dalle azioni che
avrà scelto. La distribuzione degli stati è **indotta dalla politica** e non
fissata dal mondo, e quella dell'allievo non è quella del maestro.

`````

## Il problema: gli errori si compongono

Qui sta la difficoltà, ed è una di quelle che si capiscono meglio con un
esempio che con una formula.

`````{tab} Elementare

Impari a guidare guardando un pilota bravissimo, uno che non esce mai dalla
corsia. In tutte le ore registrate l'auto sta al centro della strada: non c'è
un fotogramma in cui sia storta e vada raddrizzata.

Poi guidi tu. Al primo dosso ti sposti di venti centimetri, pochissimo. Quella
scena non l'hai mai vista: fai qualcosa di plausibile e sbagliato, e sei a
quaranta. Il posto lo conosci ancora meno, e sbagli di più. Dopo dieci secondi
sei nel fosso.

Eppure ogni tua sterzata, accanto a quelle del pilota, sbagliava di pochissimo.
A fregarti è dove ti lascia ogni sbaglio: un pezzo di strada che conosci meno,
dove il prossimo sarà più grosso. Su una curva sola non te ne accorgi; su
un'ora di autostrada è la differenza fra arrivare e uscire di strada.

Più il maestro è bravo, peggio è: chi non sbaglia mai non si trova mai a
rimediare, e non ti fa vedere l'unica cosa che ti servirà. Uno scarso peggiora
tutto, perché ti mostra come si finisce storti e intanto gli copi i vizi. Serve
il pilota bravo messo nei guai.

Allora lo fai salire accanto a te, e guidi tu. Dove ti impianti gli chiedi che
cosa avrebbe fatto lì, e la sua risposta va nel quaderno con le registrazioni
vecchie. Giro dopo giro il quaderno si riempie delle strade che percorri tu, e
si smette quando non finisci più in posti nuovi. Si chiama **DAgger**, da
*Dataset Aggregation*, «accumulare dati». E alla fine non tieni per forza il
guidatore dell'ultimo giro: provi su una strada mai fatta quello di ogni giro,
e tieni il migliore. Il conto lo paghi in ore del pilota, che deve stare lì a
rispondere: la scatola delle vecchie cassette non basta più.

Oltre un certo punto non serve. Finché le ruote sono sull'asfalto, storte
quanto vuoi, lui sa raddrizzare, e uno sbaglio isolato costa poco e si paga
subito. Se la sbandata ti pianta contro il guard rail, puoi averlo seduto
accanto quanto vuoi: da un'auto ferma fra i rottami non c'è sterzata che
riporti in corsia. Il quaderno si riempie delle situazioni giuste, e nessuno
promette che da lì si torni indietro.

C'è poi chi il pilota lo guarda dall'altro capo: invece di copiargli le mani si
chiede che cosa voglia ottenere. Si chiama **apprendimento per rinforzo
inverso**, «inverso» perché di solito da un premio si ricava un modo di
guidare, e qui si va nel verso opposto. Costa di più e regge meglio se la
strada cambia un poco: un obiettivo lo porti altrove, una collezione di
reazioni no. Il difetto è che di obiettivi che spiegano quella guida ce n'è
un'infinità. Il più sfacciato non chiede niente: se ogni viaggio vale zero, il
pilota è perfetto insieme a chiunque altro. Nemmeno pretendere di più basta.
Chiedi che tutti i viaggi possibili restino nello stesso ordine, dal migliore
al peggiore, e ne sopravvivono ancora infiniti: quelli che ripetono la stessa
cosa in un'altra unità di misura, come contare in euro invece che in centesimi,
e quelli che seminano premi lungo la strada in modo che, a viaggio finito, la
somma sia la stessa qualunque strada si sia presa. Guardando solo come guida,
sceglierne uno non si può.

`````

`````{tab} Superiore

Formalmente, la clonazione minimizza l'errore sotto la distribuzione di stati
$d_{\pi^\star}$ indotta dall'esperto, mentre al momento dell'uso l'agente vive
sotto $d_{\hat\pi}$, indotta da sé stesso. È uno **spostamento di
distribuzione**, con la particolarità di essere causato dal modello stesso:
non è il mondo che cambia, è la politica che si porta in una regione che non
conosce, e più sbaglia più ci si porta.

Il risultato che inquadra il problema è di Ross e Bagnell
{cite}`ross2010efficient`, e l'attribuzione conta perché il lavoro che tutti
citano, quello del 2011, lo riprende come premessa. Se la politica appresa ha
un tasso d'errore $\epsilon$ sotto la distribuzione dell'esperto, e il costo di
un singolo passo è limitato, su un orizzonte $T$ il costo aggiuntivo
della clonazione comportamentale cresce in generale come $O(\epsilon T^2)$: il
fattore $T$ in più rispetto all'ideale $O(\epsilon T)$ è esattamente la
composizione degli errori. Su orizzonti lunghi la differenza fra $T$ e $T^2$ è
tutta la differenza fra un sistema che funziona e uno che no.

**DAgger** (*Dataset Aggregation*), proposto da Ross, Gordon e Bagnell nel
2011 {cite}`ross2011reduction`, rimuove il termine in più con un'idea
semplice: iterare. Si addestra una politica sulle dimostrazioni, la si
esegue per raccogliere gli stati che *lei* visita, si chiede all'esperto
l'azione corretta su quegli stati, si aggiunge tutto al dataset e si
riaddestra. Ripetendo, la distribuzione di addestramento converge a quella
d'uso, e la garanzia torna lineare in $T$ a certe condizioni, che sono
nascoste in una costante e in un pedice e vanno tirate fuori tutte e due. Il
bound, dopo $N$ giri, è

$$
C(\hat\pi) \;\le\; C(\pi^\star) + u\,T\,\epsilon_N + O(1),
$$

dove $C$ è il **costo** atteso di una politica sull'orizzonte $T$, cioè un
danno da contenere e non un ritorno da massimizzare: per questo $C(\hat\pi)$
sta a sinistra della disuguaglianza (nel lavoro originale è chiamato $J$).
Quanto a $\epsilon_N$, è l'errore della migliore politica *col senno di poi* e
non l’$\epsilon$ misurato sotto la distribuzione dell'esperto: misurato sulla
media di tutte le distribuzioni di stati accumulate nei $N$ giri. Che quel
numero sia piccolo non è gratis: lo garantisce il fatto che i giri si
comportino come un algoritmo *no-regret*, e servono $N$ dell'ordine di $u\,T$
perché il resto sia davvero $O(1)$. La garanzia, poi, copre la migliore delle
politiche prodotte lungo la sequenza e non necessariamente l'ultima, ed è il
motivo per cui in pratica si sceglie su un insieme di validazione quale tenere.

Quanto a $u$, misura di quanto un singolo errore può peggiorare il
costo-per-andare dell'esperto. Nei compiti **recuperabili** $u$ è $O(1)$ e la
garanzia è
effettivamente lineare; ma se un errore porta in uno stato da cui non si torna
(l'auto ribaltata nel fosso, che nessuna sterzata riporta in corsia) $u$ può
crescere come $T$, e si
torna al quadrato. L'ipotesi va tenuta in mente davanti a un caso concreto:
uno scarto che, lasciato a sé, cresce a ogni passo, e un controllore esperto
che lo riporta verso lo zero da qualunque punto, moltiplicandolo per
$0{,}85$ ogni volta. Lì l'ipotesi è soddisfatta, e non per caso: un errore
isolato costa una quantità limitata e indipendente dall'orizzonte, e pochi giri
di DAgger bastano a tornare al livello dell'esperto. Dove invece il fosso è
un fosso vero, DAgger raccoglie comunque gli stati giusti, ma non promette che
da lì si possa tornare.

Il prezzo, poi, è che serve un esperto interrogabile durante
l'addestramento, non solo un archivio di registrazioni: e nella maggior parte
dei casi pratici quell'esperto è una persona, il che sposta il costo dai dati al
tempo umano.

La clonazione ha un parente che risolve lo stesso problema in un altro modo, e
i due si tengono distinti. Nell’**apprendimento per rinforzo inverso** non si
impara la politica, si impara la **ricompensa** che rende ottimo il
comportamento osservato, e poi la si ottimizza con i metodi dei capitoli
precedenti. È più costoso, ed è più robusto per una ragione precisa: una
ricompensa è una descrizione compatta e trasferibile dell'obiettivo, mentre
una politica è una tabella di reazioni valida solo dove è stata vista. Se
l'ambiente cambia un po’, la ricompensa regge e la politica no. È lo stesso
motivo per cui l'RLHF non si ferma alla fase supervisionata: il modello di
ricompensa addestrato sulle preferenze è, di fatto, una ricompensa inferita da
comportamento umano.

Il difetto strutturale, però, gli sta accanto fin dal lavoro che ne dà i primi
algoritmi {cite}`ng2000algorithms`: l'RL inverso, nella sua forma nuda, è **mal
posto**. Infinite funzioni di ricompensa rendono ottimo lo stesso comportamento
osservato, a cominciare da quella identicamente nulla, sotto la quale ogni
politica è ottima: l'insieme delle ricompense compatibili con una politica
osservata è un poliedro e non una retta. Nemmeno chiedendo molto di più, cioè
che l'ordinamento di *tutte* le politiche resti quello, si arriva a una
risposta sola: si arriva a una scala positiva e a un termine di *shaping
potential-based* {cite}`ng1999policy`, e non oltre. È un oggetto che il
capitolo rincontrerà: nella {doc}`sezione sull'esplorazione
<esplorazione-e-ricompensa>` si dimostra che aggiungere alla ricompensa un
termine della forma $\gamma\Phi(s')-\Phi(s)$ lascia invariata la policy ottima,
*per qualunque* $\Phi$. Là quella proprietà è una garanzia (si possono dare
aiuti senza spostare l'obiettivo); qui, letta al rovescio, è esattamente
l'ambiguità che l'RL inverso non può sciogliere. Stessa proprietà, due lati.

`````

## In pratica: che cosa succede appena si esce da ciò che il maestro ha mostrato

Quell'affermazione si può toccare con mano: è l'auto storta del racconto,
scritta in numeri, e i numeri sono pochissimi.

Di quanto siamo fuori posto lo dice un numero solo, che chiameremo $s$: zero
vuol dire dritti in mezzo alla corsia, e più cresce (in positivo o in negativo)
più siamo storti. Il sistema è **instabile**, cioè lasciato a sé peggiora invece
di rimettersi a posto: senza correzioni quel numero, a ogni passo, si moltiplica
per $1{,}25$. La correzione che diamo la chiameremo $a$, e si somma: al passo
dopo lo scarto vale $1{,}25\,s + a$.

Da qui esce il numero che conta in tutta la sezione. Il sistema, da solo,
aggiunge allo scarto un quarto di se stesso ($1{,}25 - 1 = 0{,}25$). Perché
l'auto si raddrizzi invece di storcersi ancora, la correzione deve come minimo
cancellare quell'aggiunta: deve cioè valere almeno **un quarto dello scarto**.

L'esperto è un controllore che sa cosa fare ovunque, anche lontanissimo da
zero: la sua regola è correggere in proporzione allo scarto, così da riportarlo
ogni volta all’$85\%$ di quello che era. A $s = 3$ corregge di $1{,}2$, e il
conto torna: $1{,}25 \times 3 = 3{,}75$, meno $1{,}2$ fa $2{,}55$, che è
esattamente l’$85\%$ di $3$. A $s = 20$ correggerà di $8$, e così via. E siccome
è bravo, dallo zero non si allontana mai. Poi, a
metà partita, diamo una folata di vento che l'esperto nelle sue registrazioni
non ha mai preso.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

INSTABILE, RUMORE, ORIZZONTE, RAFFICA = 1.25, 0.02, 40, 4.0

def esperto(s):
    """Un controllore vero: sa cosa fare ovunque, anche lontano da zero."""
    return -(INSTABILE - 0.85) * s

def episodio(politica, s0, g, raffica=False):
    s, stati, azioni = s0.clone(), [], []
    for t in range(ORIZZONTE):
        if raffica and t == 15:
            s = s + RAFFICA                 # una folata che l'esperto non ha mai preso
        with torch.no_grad():
            a = politica(s)
        stati.append(s.clone()); azioni.append(esperto(s))   # l'esperto etichetta
        s = INSTABILE * s + a + RUMORE * torch.randn(s.shape, generator=g)
    return torch.cat(stati), torch.cat(azioni), s

def rete():
    return nn.Sequential(nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 64),
                         nn.Tanh(), nn.Linear(64, 1))

def allena(politica, S, A, passi=2000):
    ott = torch.optim.Adam(politica.parameters(), lr=3e-3)
    for _ in range(passi):
        ott.zero_grad(); F.mse_loss(politica(S), A).backward(); ott.step()

torch.manual_seed(0)
g = torch.Generator().manual_seed(1)
avvio = lambda n: torch.randn(n, 1, generator=g) * 0.3

# le dimostrazioni: l'esperto è bravo, quindi non si allontana mai da zero
S_dim, A_dim, _ = episodio(esperto, avvio(64), g)
print(f"stati dimostrati: fra {S_dim.min():+.2f} e {S_dim.max():+.2f}")

politica = rete()
allena(politica, S_dim, A_dim)
with torch.no_grad():
    print(f"errore per singolo passo sugli stati dimostrati: "
          f"{F.mse_loss(politica(S_dim), A_dim).item():.6f}  (praticamente perfetto)")
    # fuori dalla fascia dimostrata: che cosa propone, e che cosa servirebbe?
    for x in (3.0, 4.0, 20.0):
        fuori = torch.tensor([[x]])
        print(f"a s={x:5.1f}: l'esperto direbbe {esperto(fuori).item():+7.3f}, "
              f"la clonazione dice {politica(fuori).item():+.3f} "
              f"(per rientrare servirebbe piu' di {0.25 * x:.2f})")

_, _, fe = episodio(esperto, avvio(64), g, raffica=True)
_, _, fc = episodio(politica, avvio(64), g, raffica=True)
print(f"\ndopo la folata: |stato| finale esperto {fe.abs().mean():.3f}, "
      f"clonazione {fc.abs().mean():.1f}")

# DAgger: si aggiungono le situazioni in cui e' finito L'ALLIEVO, etichettate
# dall'esperto. Sono proprio quelle che le dimostrazioni non contenevano.
S_tot, A_tot = S_dim, A_dim
for giro in (1, 2, 3):
    S_v, A_v, _ = episodio(politica, avvio(64), g, raffica=True)
    S_tot, A_tot = torch.cat([S_tot, S_v]), torch.cat([A_tot, A_v])
    allena(politica, S_tot, A_tot, passi=1500)
    _, _, f = episodio(politica, avvio(64), g, raffica=True)
    print(f"dopo il giro {giro} di DAgger: |stato| finale {f.abs().mean():.3f}")

# Controllo, e vale quanto l'esperimento: quel 74,6 è UN seme, e la folata
# gliela diamo noi. Che cosa succede su otto semi, e senza la folata?
def prova(seme):
    torch.manual_seed(seme)
    g = torch.Generator().manual_seed(seme + 1)
    avvio = lambda n: torch.randn(n, 1, generator=g) * 0.3
    S, A, _ = episodio(esperto, avvio(64), g)
    pol = rete(); allena(pol, S, A)
    _, _, e_senza = episodio(esperto, avvio(64), g)          # esperto, nessuna folata
    _, _, c_senza = episodio(pol, avvio(64), g)              # clonazione, nessuna folata
    _, _, c_con = episodio(pol, avvio(64), g, raffica=True)  # clonazione, con folata
    return (e_senza.abs().mean().item(), c_senza.abs().mean().item(),
            c_con.abs().mean().item())

esiti = [prova(s) for s in range(8)]
for nome, colonna in (("esperto, senza folata   ", 0),
                      ("clonazione, senza folata", 1),
                      ("clonazione, con folata  ", 2)):
    v = sorted(e[colonna] for e in esiti)
    print(f"{nome}: mediana {(v[3] + v[4]) / 2:.4g}, da {v[0]:.4g} a {v[-1]:.4g}")
```

I numeri raccontano la storia meglio di qualunque spiegazione. (Nelle stampe,
`|stato|` con le due stanghette vuol dire «quanto è grande lo scarto», senza
guardare se siamo storti da una parte o dall'altra.)

L'esperto vive in una fascia strettissima, fra $-1{,}00$ e $+0{,}71$. Su quella
fascia la clonazione impara **perfettamente**: l'errore per singolo passo è
$0{,}000000$, e qualunque valutazione fatta sui dati di addestramento direbbe
che il modello è impeccabile.

A $s = 3{,}0$, però, dove non è mai stata, l'esperto correggerebbe di $1{,}200$
e la clonazione propone $0{,}824$: una correzione **più debole di quasi un
terzo**.
(Da qui in poi guardiamo solo quanto è grande la correzione e non il suo segno,
perché la direzione è sempre la stessa: verso il centro della corsia.) Sembra
poco, e invece siamo sul filo. Il minimo che serve a raddrizzarsi, a $s = 3$, è
un quarto di $3$, cioè $0{,}75$: la clonazione lo copre appena.

Poco più in là non lo copre più, perché fuori dalla fascia che ha visto la rete
non cresce insieme allo scarto, si **appiattisce**: a $s = 4$ propone $0{,}904$
dove ne servirebbe $1{,}00$, e a $s = 20$ propone ancora $1{,}015$ dove ne
servirebbero $5{,}00$. Da lì in poi lo scarto cresce, e più cresce più si
allontana da ciò che la clonazione conosce. La folata, che vale $4{,}0$, la
scaraventa esattamente di là.

Il risultato: dopo la folata l'esperto torna a $0{,}071$ e **la clonazione
finisce a $74{,}6$**. Mille volte più lontano dal centro della corsia, partendo
da un errore per passo pari a zero.

Poi i tre giri di DAgger: $0{,}102$, $0{,}077$, $0{,}082$. Riportare
l'esperto a etichettare gli stati in cui era finito **l'allievo** basta a
tornare al livello del maestro. E si noti cosa è cambiato: non il modello, non
il modo di misurare l'errore, non il procedimento con cui lo si riduce. Sono
cambiati **quali situazioni stanno nel mucchio degli esempi**.

### Che cosa questo esperimento dimostra, e che cosa no

Primo, quel $74{,}6$ è **un seme**, cioè una singola ripetizione, quella che
esce dal numero da cui è partito il sorteggio interno. Rifacendo tutto da capo
con otto semi diversi, lo stato finale della clonazione dopo la folata ha
mediana $322$ e va da $76$ a $473$: il numero del racconto sta perfino **sotto**
il più mite degli otto, e la mediana è più di quattro volte più grande. La
conclusione qualitativa non cambia di una virgola (la clonazione finisce fuori
strada in tutti e otto i casi, finendo cento o mille volte più lontano
dell'esperto), ma la cifra precisa è una proprietà di quella ripetizione e non
dell'algoritmo, e per giunta quella che fa apparire il guaio
più piccolo di com'è.

Secondo, e conta di più: **senza la folata non succede niente**. Sugli stessi
otto semi, lasciata a sé, la clonazione chiude con mediana $0{,}0285$ (fra
$0{,}024$ e $0{,}031$) e l'esperto con mediana $0{,}0318$: non solo sono dello
stesso ordine, ma su questi otto semi l'allievo sta perfino un filo meglio del
maestro, che è quanto dire che la differenza è rumore. Il fosso arriva solo
quando qualcosa porta l'allievo fuori dalla fascia dimostrata, e in questo
esperimento a portarcelo è una perturbazione **esterna**, che gli diamo noi, e
che in un colpo solo lo scaraventa a quattro volte il bordo della fascia.

Nel racconto della guida, invece, fuori dalla fascia l'allievo ci arriva **da
solo**, un errore alla volta. E qui conviene contare, perché il conto è la parte
sorprendente. Le occasioni di sbagliare sono tante quanti i passi del percorso;
e ogni singolo sbaglio non si paga una volta sola, perché lascia l'allievo in
una zona che non conosce per tutti i passi che restano. Tanti inciampi
possibili, e ciascuno che si paga a lungo: le due quantità si moltiplicano fra
loro invece di sommarsi, e il danno cresce come il **quadrato** della durata del
percorso invece che in proporzione a essa. Su un tragitto dieci volte più lungo
il danno non decuplica: si moltiplica per cento.

Qui la spinta iniziale è un espediente, il modo più rapido di mettere l'allievo
dove non è mai stato per mostrare cosa succede una volta che ci si trova. Quello
che l'esperimento dimostra, e lo dimostra bene, è la seconda metà del
ragionamento: **fuori dalle situazioni che il maestro ha mostrato l'allievo
sbaglia in modo sistematico, e su un sistema instabile sbagliare in modo
sistematico è irrecuperabile.** La prima metà, cioè che a portarlo fuori bastino
i suoi stessi errori, resta un risultato teorico, e questo codice non la prova.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- La **clonazione comportamentale** è la scorciatoia più ovvia: si registra
  qualcuno che il lavoro lo sa fare, si annota «in questa situazione, questa
  mossa», e da lì in poi il problema diventa indovinare la risposta giusta,
  invece di imparare per tentativi. Stabile, sicura, parca di dati, e già
  incontrata due volte senza chiamarla per nome.
- La crepa sta in un'assunzione che nessuno dichiara: che le situazioni siano
  sempre le stesse, decise dal mondo. Non è così, perché **le situazioni che
  incontri dipendono dalle mosse che hai fatto**, e le mosse dell'allievo non
  sono quelle del maestro.
- Da qui la **composizione degli errori**: un errore piccolo ti porta in un
  posto che conosci meno, dove sbagli di più, che ti porta ancora più lontano.
  L'errore non si somma, si compone, e su un percorso lungo è la differenza fra
  un sistema che funziona e uno che no.
- Il paradosso da ricordare: **più l'esperto è bravo, meno insegna a
  rimediare**, perché non si trova mai nella condizione di doverlo fare. Il
  pilota che non esce mai di corsia non ti fa mai vedere come si raddrizza
  l'auto.
- Il rimedio si chiama **DAgger**: far provare l'allievo, guardare dove va a
  finire, e chiedere al maestro cosa avrebbe fatto *lì*. Non cambia il modello e
  non cambia il modo di allenarlo: cambia quali situazioni finiscono nel mucchio
  degli esempi. Il prezzo è che serve un maestro disponibile a rispondere, non
  soltanto un archivio di registrazioni; e funziona finché dallo sbaglio si può
  tornare, perché dall'auto finita contro il guard rail non riporta indietro
  nessuno.
- C'è un'altra strada: invece della strategia, imparare la **ricompensa**, cioè
  che cosa l'esperto stesse cercando di ottenere. Costa di più e regge meglio se
  il mondo cambia un po’, perché descrive l'obiettivo e non solo le reazioni. Ha
  un difetto suo, però: di obiettivi che spiegano lo stesso comportamento ce n'è
  un'infinità, e distinguerli guardando solo il comportamento non si può.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- La **clonazione comportamentale** trasforma il controllo in apprendimento
  supervisionato: coppie (situazione, azione dell'esperto), e si minimizza
  l'errore. Stabile, efficiente nei dati, sicura, e già usata due volte nel
  libro senza il suo nome (la policy iniziale di AlphaGo, la fase supervisionata
  che precede l'RLHF).
- L'assunzione che salta è quella di dati **i.i.d.**: la distribuzione degli
  stati è **indotta dalla politica** e non fissata dal mondo, e quella
  dell'allievo non è quella del maestro.
- Da qui la **composizione degli errori**: un errore piccolo porta in uno stato
  poco familiare, dove l'errore è più grande, e così via. Il costo cresce come
  $O(\epsilon T^2)$ invece di $O(\epsilon T)$.
- Il paradosso da ricordare: **più l'esperto è bravo, meno insegna a
  rimediare**, perché non si trova mai nella condizione di doverlo fare.
- **DAgger** rimuove il termine in più iterando: esegui la politica, chiedi
  all'esperto cosa fare **negli stati che lei visita**, aggiungi, riaddestra.
  Serve un esperto interrogabile, non solo un archivio; e la garanzia lineare
  vale a patto che il compito sia **recuperabile**, altrimenti la costante $u$
  del bound cresce con $T$ e si torna al quadrato.
- L’**RL inverso** risolve lo stesso problema per un'altra strada: impara la
  **ricompensa** invece della politica. Costa di più ed è più trasferibile,
  perché una ricompensa descrive l'obiettivo mentre una politica descrive solo
  delle reazioni. È la ragione per cui l'RLHF non si ferma alla fase
  supervisionata. È però **mal posto**: dalla sola politica osservata le
  ricompense compatibili formano un poliedro (la nulla compresa), e anche
  imponendo di preservare l'ordinamento di *tutte* le politiche non si va oltre
  una scala positiva e un termine di shaping potential-based.
```
`````
