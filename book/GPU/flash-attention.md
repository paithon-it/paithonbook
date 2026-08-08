# Flash Attention: l'attenzione che non spreca memoria

Chiedere a un modello di riassumere un romanzo intero, o di rispondere su un
contratto di cento pagine, fino a pochi anni fa era impensabile, e non per una
carenza di intelligenza. L'ostacolo era idraulico: una matrice che cresce con
il *quadrato* della lunghezza del testo. Raddoppia le parole e quella matrice
quadruplica; moltiplicale per dieci e diventa cento volte più grande. A un
certo punto non ci sta più nella memoria della GPU, e anche quando ci sta,
spostarla avanti e indietro costa così tanto tempo da rendere tutto
insopportabilmente lento. Questa sezione racconta l'idea (sorprendentemente
semplice nella sostanza) che ha fatto saltare quel muro e ha reso possibili i
contesti lunghi di oggi.

Nel capitolo sui **Transformer** abbiamo visto *cosa* fa l'attenzione: per
ogni posizione confronta una *query* con tutte le *key*, normalizza i punteggi
con una softmax e usa i pesi per mediare i *value*, in una riga sola,
$\text{Attention}(Q,K,V) = \text{softmax}\!\big(QK^\top/\sqrt{d_k}\big)V$. Lì
il problema era *quale* informazione l'attenzione raccoglie. Qui il problema è
un altro, tutto hardware: *come* si esegue quel conto senza affogare nel
traffico di memoria. Non ri-deriviamo l'attenzione (diamo per acquisito il
capitolo sui Transformer) e ci concentriamo sul *come*.

## Il problema è la memoria, non i conti

Il primo istinto è pensare che l'attenzione sia lenta perché fa *tanti conti*.
È vero solo a metà. Il vero collo di bottiglia, come quasi sempre su una GPU,
non è il calcolo: è il **movimento dei dati** (la stessa lezione della
gerarchia di memoria e del roofline delle sezioni precedenti).

`````{tab} Elementare
Immagina di dover confrontare ogni parola di un testo con ogni altra parola.
Con mille parole, sono un milione di confronti: una tabella di mille righe per
mille colonne. Fin qui, tanti conti ma niente di drammatico. Il guaio è *dove*
metti quella tabella. È troppo grande per il tavolo di lavoro veloce (la
memoria on-chip), quindi la GPU la scrive nel magazzino lontano (la memoria
grande e lenta) e poi deve tornare a prenderla per fare il passo successivo,
la softmax, e poi *di nuovo* per l'ultima moltiplicazione. Tre spedizioni al
magazzino per una tabella enorme che, alla fine, non serviva nemmeno tenere:
era solo un passaggio intermedio. È come impastare tutta la pasta del mondo,
stenderla su un tavolo lungo un chilometro solo per tagliarla, e correre
avanti e indietro per tutta la lunghezza a ogni operazione. Il tempo non se ne
va nel taglio: se ne va nella corsa.
`````

`````{tab} Superiore
Per una sequenza di lunghezza $N$ e teste di dimensione $d_k$, l'attenzione
materializza due matrici $N \times N$: i punteggi $S = QK^\top/\sqrt{d_k}$ e i
pesi $P = \text{softmax}(S)$. Il calcolo è $O(N^2 d_k)$ FLOP, ma il dato che
uccide le prestazioni è la **memoria**: $S$ e $P$ occupano $O(N^2)$ byte e
vengono scritte e rilette dalla HBM più volte (produci $S$, la rileggi per la
softmax, rileggi $P$ per il prodotto con $V$). Un numero concreto: con
$N = 8192$, una sola matrice $S$ ha $N^2 \approx 67$ milioni di elementi; in
`float16` (2 byte) sono circa $134$ MB (*per testa, per strato*). La memoria
cresce quadraticamente, e con essa il traffico verso la HBM.

Sul roofline questa è l'operazione tipicamente **memory-bound**: le parti pesanti
del calcolo (i due matmul $QK^\top$ e $PV$) hanno buona intensità aritmetica, ma
in mezzo stanno operazioni a bassissima intensità (gli esponenziali e le
riduzioni per riga della softmax, le scritture e riletture della matrice
$N \times N$) che trascinano l'intera operazione
contro il tetto di banda. Non serve una GPU più potente nei FLOP: serve *non
spostare* quei byte.
`````

## L'idea: lavorare a tessere, mai scrivere la matrice

La svolta arriva nel 2022 da Tri Dao e colleghi con **FlashAttention**
{cite}`dao2022flashattention`. La loro osservazione è che la matrice
$N \times N$ è solo un *intermedio*: alla fine ci serve l'output, non la
tabella dei punteggi. E allora perché scriverla? L'algoritmo è **IO-aware**:
ottimizza il movimento dei dati, non i conti, e (dettaglio cruciale) dà il
**risultato esatto**, non un'approssimazione.

Due ingredienti lo rendono possibile ({numref}`fig-flash-attention`): il
**tiling** di $Q$, $K$, $V$ (lo stesso «carica una tessera, riusala» del GEMM
della sezione precedente) e la **online softmax**, che permette di
normalizzare i punteggi *a pezzi* invece che tutti insieme.

```{figure} ../figures/flash-attention-tiling.svg
:name: fig-flash-attention
:alt: "A sinistra la matrice dei punteggi S uguale Q per K trasposto, N per N, disegnata come griglia e barrata da una grande X: la matrice che FlashAttention non scrive mai nella memoria HBM. A destra lo schema: una shared memory on-chip tiene un tile fisso di Q e un blocco corrente di K e V; sotto, i blocchi di K e V scorrono uno per volta dalla HBM verso la shared memory; un accumulatore aggiorna a ogni blocco l'output O e le due statistiche del softmax, il massimo corrente m e la somma corrente l; alla fine l'uscita è O diviso l, ed è esatta."
:width: 90%

FlashAttention non materializza mai la matrice $N \times N$: tiene un tile di
$Q$ in shared memory e vi fa scorrere i blocchi di $K,V$ uno per volta,
aggiornando a ogni passo l'output $O$ e le statistiche del softmax (il massimo
corrente $m$ e la somma corrente $l$).
```

`````{tab} Elementare
Il trucco è non costruire mai la tabella gigante. Tieni ferma sul tavolo di
lavoro una manciata di query (una *tessera* di $Q$) e fai scorrere il testo
sorgente a blocchetti: prendi le prime chiavi, calcoli i loro punteggi,
aggiorni il risultato; butti via quel blocchetto, prendi il successivo, e così
via fino alla fine. Sul tavolo, in ogni istante, c'è solo un pezzetto piccolo:
quello che stai usando *adesso*. La tabella da un milione di caselle non viene
mai scritta per intero da nessuna parte: esiste solo un blocchetto alla volta,
e sparisce appena hai finito di usarlo. Meno viaggi al magazzino, stesso
risultato.

C'è però un'insidia da risolvere: come fai a calcolare le *percentuali* (la
softmax trasforma i punteggi in pesi che sommano a 100%) se non hai ancora visto
tutti i punteggi? Per fare una percentuale ti serve il totale, e il totale lo
conosci solo alla fine. La risposta è la *online softmax* del prossimo passaggio.
`````

`````{tab} Superiore
Formalmente, si spezzano $Q$, $K$, $V$ in blocchi di righe. Per un blocco di
query, si itera sui blocchi $(K_j, V_j)$: si carica $K_j, V_j$ in **shared
memory**, si calcola il tile di punteggi $S_j = Q K_j^\top/\sqrt{d_k}$, e si
aggiorna l'output *sul posto*, senza mai scrivere l'intera matrice $S$ in HBM.
(Quest'ordine dei cicli, con il blocco di query fermo e $K,V$ che scorrono, è
quello reso canonico dalla seconda versione dell'algoritmo, che incontreremo a
breve; l'articolo del 2022 li annidava al contrario, ma l'idea non cambia.) La
memoria on-chip trattiene solo i tile correnti; la HBM vede scorrere $K,V$ una
volta per ogni blocco di query, e la matrice $S$ mai. Il costo in FLOP resta
$O(N^2 d_k)$ (non tocchiamo i conti), ma la **memoria extra** scende da
$O(N^2)$ a $O(N)$: da scrivere restano solo l'output e le statistiche di riga.
Anche il traffico verso la HBM crolla: il paper lo conta in
$\Theta(N^2 d_k^2 / M)$ accessi, dove $M$ è la taglia della memoria on-chip,
contro il $\Theta(N d_k + N^2)$ dell'attenzione standard. Resta quadratico in
$N$, ma diviso per un fattore $M/d_k^2$ che, con una SRAM on-chip di qualche
decina di migliaia di elementi per SM, vale qualche unità con $d_k = 128$ e
più di una decina con $d_k = 64$ (il paper si limita a dire che $d_k^2$ è
molte volte più piccolo di $M$). È un fattore che su un carico memory-bound
conta. È l'idea del tiling in shared memory del GEMM, applicata
all'attenzione: caricare una volta, riusare in tanti, non tornare al
magazzino.

Il nodo tecnico è che la softmax *non* è elemento-per-elemento: normalizza per
righe, e la normalizzazione richiede in teoria di aver già visto tutti i
punteggi della riga. Scorrere $K$ a blocchi significa vedere i punteggi un
pezzo per volta, e qui entra la online softmax.
`````

### La online softmax, con i numeri

Il perno di tutto è calcolare una softmax vedendo i punteggi **a blocchi**, senza
mai averli tutti sotto gli occhi insieme, e ottenendo comunque il risultato
esatto. Serve tenere solo due numeri di riepilogo: il **massimo corrente** $m$ e
la **somma corrente** $l$.

`````{tab} Elementare
È lo stesso gesto di chi tiene una media aggiornata senza scrivere ogni singolo
valore. La maestra deve fare la media di trecento compiti ma sul banco ne stanno
dieci per volta: tiene due foglietti, «somma finora» e «quanti finora», li
aggiorna a ogni mucchietto, e alla fine divide. Il risultato è identico a quello
che otterrebbe stendendo tutti i trecento compiti sul pavimento in una volta
sola. La online softmax fa così con i punteggi dell'attenzione: aggiorna a ogni
blocco il totale (e il valore più grande visto finora, che serve a tenere i conti
in scala), e alla fine ottiene *esattamente* le stesse percentuali del calcolo in
un colpo unico. Nessuna approssimazione: solo la stessa somma, fatta a rate.
`````

`````{tab} Superiore
Facciamo il conto a mano su una riga di quattro punteggi
$s = (1, 3, 2, 4)$ (i valori di $QK^\top/\sqrt{d_k}$ per una query contro quattro
key). Il calcolo *in un colpo solo*, con la solita stabilizzazione che sottrae il
massimo per non far esplodere gli esponenziali:

$$
m = \max(s) = 4, \qquad
l = \sum_i e^{s_i - m} = e^{-3}+e^{-1}+e^{-2}+e^{0} \approx 1{,}553,
$$

da cui i pesi softmax $(0{,}032,\ 0{,}237,\ 0{,}087,\ 0{,}644)$.

Ora *a blocchi di due*, $[1,3]$ poi $[2,4]$, tenendo aggiornati $m$ e $l$:

- **Blocco 1** $[1,3]$:  $\ m_1 = 3$,  $\ l_1 = e^{1-3}+e^{3-3} = e^{-2}+1 \approx 1{,}135$.
- **Blocco 2** $[2,4]$:  il massimo del blocco è $4$, quindi $m_2 = \max(3,4) = 4$.
  Il vecchio totale va **ri-scalato** al nuovo massimo con il fattore di
  correzione $\alpha = e^{m_1 - m_2} = e^{-1} \approx 0{,}368$:

$$
l_2 = \alpha\, l_1 + \big(e^{2-4}+e^{4-4}\big)
    = 0{,}368 \cdot 1{,}135 + e^{-2}+1 \approx 1{,}553.
$$

Il totale $l_2 \approx 1{,}553$ coincide **esattamente** con la somma calcolata in
un colpo solo: la online softmax dà gli stessi pesi. In generale, arrivando un
nuovo blocco con massimo locale $\tilde m$, le regole di aggiornamento sono

$$
m^{\text{new}} = \max(m, \tilde m), \quad
l^{\text{new}} = e^{\,m - m^{\text{new}}}\, l + \!\sum_{i \in \text{blocco}}\! e^{\,s_i - m^{\text{new}}}, \quad
O^{\text{new}} = e^{\,m - m^{\text{new}}}\, O + \!\sum_{i \in \text{blocco}}\! e^{\,s_i - m^{\text{new}}}\, v_i,
$$

dove $O$ è l'output accumulato (somma pesata dei $v_i$) e il fattore
$e^{\,m - m^{\text{new}}}$ corregge ciò che avevamo già sommato quando compare un
massimo nuovo; alla fine si divide, $O \leftarrow O/l$. Tutto qui: due scalari di
stato per riga, e la matrice $N \times N$ non viene mai scritta.
`````

## Cosa si guadagna (e cosa costa)

Il risultato è netto: l'attenzione passa da $O(N^2)$ a $O(N)$ di memoria,
senza cambiare di una virgola il valore calcolato. A sequenze corte il
guadagno è modesto, ma cresce con $N$, ed è proprio sulle sequenze lunghe,
dove la vecchia attenzione andava in *out of memory* o rallentava fino a
fermarsi, che FlashAttention cambia le carte in tavola: è ciò che ha reso
pratici i contesti di decine o centinaia di migliaia di token. Una versione
successiva, **FlashAttention-2** {cite}`dao2023flashattention2`, spreme ancora
di più l'hardware ripartendo meglio il lavoro tra i gruppi di thread e
limitando le operazioni più lente (quelle diverse dalle moltiplicazioni di
matrici, che i tensor core non sanno accelerare) con un altro sostanziale
guadagno di velocità, nell'ordine del raddoppio rispetto alla prima versione.

Onestà, però: quello che in queste pagine sta in un'idea semplice, nel codice
è un kernel notoriamente complicato (indici, gestione della shared memory,
casi limite della maschera causale). Non è codice che si scrive a mano per un
progetto normale, ed è giusto così. In PyTorch lo usi senza nemmeno saperlo:
la funzione `scaled_dot_product_attention` seleziona da sé il backend migliore
disponibile e, su GPU adatte, quello è proprio FlashAttention.

```python
import torch
import torch.nn.functional as F

# Q, K, V: (batch, teste, N, d_k)
Q = torch.randn(2, 8, 4096, 64, device="cuda", dtype=torch.float16)
K = torch.randn_like(Q)
V = torch.randn_like(Q)

# PyTorch sceglie da sé il kernel: su GPU recenti, il backend FlashAttention.
# is_causal=True applica la maschera causale senza materializzarla.
O = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
print(O.shape)  # torch.Size([2, 8, 4096, 64])
```

Una riga di libreria, e sotto gira il kernel che abbiamo appena raccontato. È il
modo giusto di usarlo: capirne l'idea per sapere *quando* e *perché* aiuta, e
lasciarne l'implementazione a chi la mantiene ottimizzata generazione dopo
generazione.

## La frontiera: nascondere il movimento dei dati

FlashAttention è l'esempio più limpido di un filo conduttore che attraversa
tutto questo capitolo, ed è il modo migliore per chiuderlo dal lato
dell'hardware: **la storia della velocità sulle GPU è la storia di come
nascondere il movimento dei dati**. Ogni tecnica che abbiamo incontrato
(coalescenza, tiling in shared memory, fusione dei kernel, e ora l'attenzione
IO-aware) è una variazione sullo stesso tema: fare più conti per ogni byte
spostato, e tenere il byte il più vicino possibile ai core.

I kernel più veloci di oggi portano questa idea ancora più in là. Restando al
livello concettuale (niente istruzioni di basso livello) le leve sono tre.

`````{tab} Elementare
Pensa a una catena di montaggio ben organizzata. Nelle GPU più recenti, mentre
un gruppo di operai lavora sui pezzi che ha già sul banco, un *altro* gruppo è
già andato a prendere i pezzi successivi dal magazzino: quando i primi
finiscono, il materiale nuovo è lì pronto, e nessuno resta mai fermo ad
aspettare. È il **movimento asincrono** dei dati: la copia dal magazzino e il
lavoro sul banco avvengono *nello stesso momento*, sovrapposti, invece che uno
dopo l'altro. E i ruoli si specializzano: c'è chi fa solo il portapacchi e chi
solo il montaggio, come in una catena vera, perché un operaio dedicato a un
compito lo fa meglio di uno che salta di continuo da un lavoro all'altro. Più
i tavoli di lavoro (i *tensor core*) diventano potenti, più questo conta: se i
pezzi non arrivano in tempo, il tavolo più veloce del mondo resta a girarsi i
pollici. Tutta l'arte sta nel far arrivare i pezzi *mentre* si lavora, così
che il tavolo non si fermi mai.
`````

`````{tab} Superiore
Tre direzioni, tutte volte a *nascondere* la latenza del movimento dati dietro il
calcolo:

- **Movimento asincrono dei dati.** Le GPU recenti (dall'architettura Hopper
  in poi) hanno unità dedicate (come il *Tensor Memory Accelerator*, TMA) che
  copiano tessere dalla HBM alla shared memory *in parallelo* al calcolo sui
  tensor core, sovrapponendo trasferimento ed esecuzione. Il kernel non
  aspetta i dati: lavora sul tile corrente mentre il prossimo è già in
  viaggio.
- **Tensor core sempre più potenti e formati più stretti.** Le unità di matmul
  crescono in throughput di generazione in generazione (Hopper, poi Blackwell)
  e guadagnano formati numerici più compatti, fino a **FP8** (8 bit), che
  dimezzano ancora i byte da spostare, nella stessa logica della precisione
  mista vista nella sezione «Prestazioni e scala». Ma più i tensor core sono
  veloci, più è facile ritrovarsi memory-bound: il ginocchio del roofline si
  sposta a destra, e la partita torna a giocarsi sui byte.
- **Warp specialization.** Invece di far fare a ogni warp un po' di tutto, gli si
  assegnano *ruoli*: alcuni warp fanno solo da *producer* (caricano i dati dalla
  HBM), altri da *consumer* (calcolano sui tensor core), coordinati come i
  reparti di una catena di montaggio. La specializzazione tiene le unità di
  calcolo sempre rifornite e i canali di memoria sempre occupati.

Sono le tecniche con cui sono scritte le versioni più recenti dei kernel di
attenzione, che portano l'idea di FlashAttention fin sul silicio più nuovo.
Chi vuole seguirle fino in fondo (dal TMA alla warp specialization, fino alle
generazioni più recenti di FlashAttention) trova una trattazione avanzata nel
corso *Modern GPU Programming for MLSys* di mlc.ai. Il messaggio, però, resta
quello con cui abbiamo aperto il capitolo: le migliaia di core semplici sono
la parte facile; l'ingegneria vera è tenerle sfamate.
`````

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Per confrontare ogni parola con ogni altra, l'attenzione costruisce una
  tabella grande quanto il testo per il testo: raddoppia le parole e la tabella
  quadruplica. Il tempo però non se ne va nei conti: se ne va nei **viaggi** fra
  il magazzino lento e il tavolo di lavoro veloce.
- **FlashAttention** {cite}`dao2022flashattention` quella tabella non la scrive
  mai: tiene ferma sul tavolo una manciata di parole e fa scorrere le altre a
  **blocchetti**, uno per volta, buttando via ogni blocchetto appena usato. Il
  risultato non è un'approssimazione: è lo stesso identico numero di prima.
- A rendere possibile il lavoro a blocchetti è la **online softmax**, il gesto
  della maestra che corregge dieci compiti per volta: due foglietti (il totale
  finora e il valore più grande visto finora), aggiornati a ogni mucchietto,
  e alla fine le stesse percentuali del calcolo in un colpo unico.
- Il guadagno: la memoria non cresce più con il quadrato della lunghezza del
  testo ma in proporzione ad essa, e sulle sequenze lunghe (dove prima la GPU
  si fermava per memoria esaurita) il salto è grande; è la ragione per cui oggi
  si può dare in pasto a un modello un contratto di cento pagine. Una seconda
  versione, **FlashAttention-2** {cite}`dao2023flashattention2`, ripartisce
  ancora meglio il lavoro. In PyTorch basta chiamare
  `scaled_dot_product_attention`.
- I kernel più veloci di oggi (i dati che viaggiano dal magazzino *mentre* si
  lavora, tavoli di lavoro sempre più potenti che usano numeri più corti, operai
  con ruoli fissi fra chi porta i pezzi e chi li monta) girano tutti attorno
  alla stessa idea: **nascondere il movimento dei dati** dietro il calcolo, così
  che nessuno resti fermo ad aspettare.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- L'attenzione materializza due matrici $N \times N$
  ($S = QK^\top/\sqrt{d_k}$ e
  $P = \text{softmax}(S)$): $O(N^2)$ memoria e traffico HBM.
  Il collo di bottiglia è la **memoria**, non i FLOP: è **memory-bound**.
- **FlashAttention** {cite}`dao2022flashattention` è **IO-aware**: con il
  **tiling** di $Q,K,V$ in shared memory e la
  **online softmax** non scrive mai la matrice $N \times N$ in HBM. Il risultato
  è **esatto**, non approssimato.
- La **online softmax** normalizza i punteggi a blocchi tenendo due scalari di
  stato (massimo corrente $m$ e somma corrente $l$) e ri-scalando ciò che ha
  già sommato quando compare un massimo nuovo: dà gli stessi pesi del calcolo
  in un colpo solo.
- Il guadagno: memoria da $O(N^2)$ a $O(N)$, grande accelerazione a sequenze
  lunghe, contesti lunghi resi pratici. **FlashAttention-2**
  {cite}`dao2023flashattention2` migliora ancora la ripartizione del lavoro. In
  PyTorch lo si usa via `scaled_dot_product_attention`.
- La frontiera dei kernel veloci (movimento asincrono dei dati con TMA, tensor
  core e formati come FP8, **warp specialization**) è tutta una variazione sullo
  stesso tema: **nascondere il movimento dei dati** dietro il calcolo.
```
`````
