# Transformer: quando l'attenzione basta

Nell'estate del 2017 otto ricercatori di Google pubblicano un paper dal titolo
che suona come una battuta: *Attention Is All You Need*
{cite}`vaswani2017attention`; "l'attenzione è tutto ciò che serve", eco del
ritornello dei Beatles. Dentro c'è un'architettura di rete neurale nuova, il
**Transformer**, che fa una scommessa radicale: per capire il linguaggio non
servono né la ricorrenza delle RNN né le convoluzioni; basta il meccanismo di
**attenzione**, usato fino ad allora come accessorio. La scommessa è vinta
oltre ogni previsione: oggi il Transformer è la base di quasi tutti i grandi
modelli linguistici, e quella "T" è la stessa che trovi nel nome di GPT e di
ChatGPT.

## Il problema: leggere una frase tutta insieme

I modelli che abbiamo incontrato nel capitolo sul Natural Language Processing
(RNN e LSTM) leggono il testo una parola alla volta, portandosi dietro un
riassunto di quel che è venuto prima. Funziona, ma con due difetti
strutturali.

`````{tab} Elementare
Immagina di leggere un romanzo attraverso una fessura che mostra una parola
alla volta, dovendo tenere tutto il resto a memoria. Dopo dieci pagine, quanto
ricordi della prima? È il problema delle reti ricorrenti: sui testi lunghi il
ricordo dell'inizio sbiadisce. E c'è un secondo problema: se puoi leggere solo
una parola alla volta, non puoi farti aiutare; cento amici non leggono un
libro più in fretta di te se il libro va comunque letto in fila.

Il Transformer rompe la fessura: guarda **tutta la frase insieme**, e per ogni
parola decide a quali altre parole prestare attenzione. In "Il gatto nero
salta sul muro", mentre elabora "salta" può guardare direttamente "gatto" (chi
è che salta?) senza passare per un riassunto sbiadito. E siccome ogni parola
viene elaborata insieme alle altre, il lavoro si può dividere: i cento amici
servono, eccome.
`````

`````{tab} Superiore
Nelle RNN l'informazione tra due parole distanti $n$ posizioni attraversa
$O(n)$ passaggi di stato: il segnale si degrada (gradiente che svanisce, come
visto nel capitolo sulle reti neurali) e le dipendenze lunghe si perdono,
problema che LSTM e GRU mitigano ma non eliminano. Inoltre la ricorrenza è
intrinsecamente **sequenziale**: il passo $t$ richiede il passo $t-1$, e
l'hardware parallelo (le GPU) resta sottoutilizzato in addestramento.

Nel Transformer la **self-attention** collega ogni coppia di posizioni in un
solo passo, lunghezza di cammino $O(1)$, e l'elaborazione di tutte le
posizioni è un prodotto tra matrici, parallelizzabile per costruzione. È
questa seconda proprietà, più ancora della prima, ad aver cambiato la scala
dei modelli: addestrare su corpora enormi è diventato una questione di
hardware, non di architettura. Il prezzo è un costo quadratico nella lunghezza
della sequenza, di cui parleremo nella sezione sui confronti.
`````

## Che cosa troverai in questo capitolo

Il capitolo segue la scia del paper del 2017. Prima il **meccanismo di
attenzione**: cos'è, come si calcola, perché funziona. Poi l'**architettura**
completa (encoder, decoder, positional encoding), cioè come i blocchi di
attenzione si montano in una rete vera. Un confronto onesto con i **modelli
precedenti** (RNN, LSTM, GRU), inclusi i punti dove il Transformer è più
debole. Poi gli **esempi pratici**, le famiglie di modelli (GPT, BERT, T5),
l'estensione a immagini e **multimodalità**. Da lì, tre approfondimenti sul
presente: i **grandi modelli linguistici** (le leggi di scala, come si genera
il testo davvero), il **post-training** (istruzioni, preferenze umane,
allineamento: come un modello di linguaggio diventa un assistente) e il
**retrieval** (cercare per rispondere: RAG). Chiude uno sguardo alle
**tendenze**, con i limiti, che non mancano.

Una nota di metodo, coerente col resto del libro: i Transformer sono
importanti, ma non sono magia. Sono algebra lineare (matrici, prodotti
scalari, softmax) organizzata in un modo particolarmente felice. Tutto ciò che
serve per capirli lo abbiamo già costruito nei capitoli precedenti.

```{admonition} Da ricordare
:class: important
- Il **Transformer** (Vaswani et al., 2017, *Attention Is All You Need*)
  elimina ricorrenza e convoluzioni: l'intera architettura si regge sul
  meccanismo di **attenzione**.
- Rispetto alle RNN risolve due problemi: le **dipendenze lunghe** (ogni
  parola vede direttamente ogni altra) e la **parallelizzazione**
  (l'addestramento sfrutta appieno le GPU).
- La capacità di scalare con dati e parametri ne ha fatto la base dei grandi
  modelli linguistici: la "T" di GPT, BERT e ChatGPT.
```
