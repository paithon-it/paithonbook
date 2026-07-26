/**
 * PAITHON BOOK — FORMATO
 *
 * Tre funzioni. Le prime due sono ispirate a come si comportano i libri
 * tecnici costruiti con Material for MkDocs (l'esempio di riferimento è
 * learnpytorch.io); la terza è di questo libro, che spiega ogni concetto su
 * due livelli:
 *
 *   1. la scheda del repository in cima alla sidebar mostra stelle e fork;
 *   2. selezionando un pezzo di testo compaiono due azioni — aprirlo con
 *      l'assistente del magazine, o segnalarlo con una issue già compilata;
 *   3. l'interruttore in barra decide il livello di lettura (Elementare o
 *      Superiore) di tutto il libro, mentre l'apertura di una singola tab
 *      resta locale.
 *
 * Nessuna dipendenza esterna. Se GitHub non risponde, o se il browser non
 * espone la selezione, ogni pezzo si spegne in silenzio per conto suo.
 */

(function () {
  'use strict';

  // Sphinx emette due volte i file di `html_js_files` (accade anche a
  // custom.js): senza questa guardia si otterrebbero due pillole e due
  // chiamate all'API di GitHub.
  if (window.__ptFormatoCaricato) return;
  window.__ptFormatoCaricato = true;

  // ===== 1. STELLE E FORK NELLA SCHEDA DEL REPOSITORY =====
  // L'API pubblica di GitHub concede 60 richieste all'ora per IP: il
  // risultato resta in cache per sei ore, così una sessione di lettura fa
  // una sola chiamata invece di una per pagina.
  const CACHE_KEY = 'pt-repo-stats';
  const CACHE_TTL = 6 * 60 * 60 * 1000;

  function formattaNumero(n) {
    if (typeof n !== 'number' || !isFinite(n)) return null;
    if (n >= 10000) return (n / 1000).toFixed(0) + 'k';
    if (n >= 1000) return (n / 1000).toFixed(1).replace('.', ',') + 'k';
    return String(n);
  }

  function mostraStatistiche(card, dati) {
    const stelle = formattaNumero(dati.stars);
    const fork = formattaNumero(dati.forks);
    if (stelle === null && fork === null) return;

    const box = card.querySelector('[data-pt-stats]');
    if (!box) return;

    const slotStelle = box.querySelector('[data-pt-stat="stars"]');
    const slotFork = box.querySelector('[data-pt-stat="forks"]');
    if (slotStelle) slotStelle.textContent = stelle !== null ? stelle : '–';
    if (slotFork) slotFork.textContent = fork !== null ? fork : '–';
    box.hidden = false;
  }

  function statisticheRepository() {
    const card = document.querySelector('.pt-repo-card[data-pt-repo]');
    if (!card) return;

    const slug = card.dataset.ptRepo;
    if (!slug || slug.indexOf('/') === -1) return;

    // Cache locale, se ancora fresca e riferita a questo stesso repository.
    try {
      const grezzo = localStorage.getItem(CACHE_KEY);
      if (grezzo) {
        const cache = JSON.parse(grezzo);
        if (cache.slug === slug && (Date.now() - cache.t) < CACHE_TTL) {
          mostraStatistiche(card, cache);
          return;
        }
      }
    } catch (e) { /* localStorage non disponibile: si procede con la rete */ }

    fetch('https://api.github.com/repos/' + slug, {
      headers: { Accept: 'application/vnd.github+json' }
    })
      .then(function (r) { return r.ok ? r.json() : Promise.reject(r.status); })
      .then(function (j) {
        const dati = {
          slug: slug,
          stars: j.stargazers_count,
          forks: j.forks_count,
          t: Date.now()
        };
        try { localStorage.setItem(CACHE_KEY, JSON.stringify(dati)); } catch (e) {}
        mostraStatistiche(card, dati);
      })
      .catch(function () { /* niente numeri: la scheda resta un link e basta */ });
  }

  // ===== 2. SELEZIONA UN TESTO → APRI UNA ISSUE =====
  const MIN_CARATTERI = 12;   // sotto questa soglia è quasi sempre un click
  const MAX_CITAZIONE = 600;  // gli URL lunghi vengono troncati dai browser

  function meta(nome) {
    const el = document.querySelector('meta[name="' + nome + '"]');
    return el ? el.getAttribute('content') : '';
  }

  // ATTENZIONE all'ordine: Sphinx inserisce questo script nel <head> PRIMA
  // dei <meta> di head_custom.html, quindi al momento in cui il file viene
  // eseguito quei tag non esistono ancora. Le coordinate vanno lette al
  // DOMContentLoaded (da `avvia`), non qui: leggerle subito darebbe stringhe
  // vuote e il pulsante non verrebbe mai creato.
  let REPO = '';
  let RAMO = 'main';
  let SORGENTE = '';
  let LETTURA = '';

  function leggiCoordinate() {
    REPO = (meta('pt-repo-url') || '').replace(/\/+$/, '');
    RAMO = meta('pt-repo-branch') || 'main';
    SORGENTE = meta('pt-page-source') || '';
    // Pagina del magazine che raccoglie il passaggio e ci apre l'assistente.
    // Vuota = la voce "Chiedi all'AI" non compare affatto.
    LETTURA = meta('pt-lettura-url') || '';
  }

  function areaContenuto() {
    return document.querySelector('article.bd-article') ||
           document.querySelector('article') ||
           document.querySelector('main');
  }

  function titoloPagina() {
    const h1 = document.querySelector('article h1, main h1');
    if (h1) return h1.textContent.replace(/[¶#]\s*$/, '').trim();
    return (document.title || '').split('—')[0].trim();
  }

  // Risale dal nodo selezionato alla <section> che lo contiene, per citare
  // titolo e ancora del paragrafo esatto invece che della sola pagina.
  function sezioneDi(nodo) {
    let el = nodo.nodeType === 1 ? nodo : nodo.parentElement;
    while (el && el !== document.body) {
      if (el.tagName === 'SECTION' && el.id) {
        const h = el.querySelector('h1, h2, h3, h4, h5, h6');
        return {
          ancora: el.id,
          titolo: h ? h.textContent.replace(/[¶#]\s*$/, '').trim() : ''
        };
      }
      el = el.parentElement;
    }
    return { ancora: '', titolo: '' };
  }

  function tronca(testo, max) {
    return testo.length > max ? testo.slice(0, max).trimEnd() + '…' : testo;
  }

  function urlIssue(citazione, sezione) {
    const pagina = titoloPagina();
    const url = window.location.origin + window.location.pathname +
                (sezione.ancora ? '#' + sezione.ancora : '');

    const titolo = '[libro] ' + (pagina || 'segnalazione') +
                   (sezione.titolo ? ' — ' + sezione.titolo : '');

    const corpo = [
      '### Passaggio segnalato',
      '',
      '> ' + tronca(citazione, MAX_CITAZIONE).split('\n').join('\n> '),
      '',
      '### Dove',
      '',
      '- Pagina: ' + url,
      sezione.titolo ? '- Sezione: **' + sezione.titolo + '**' : null,
      SORGENTE ? '- Sorgente: [`' + SORGENTE + '`](' + REPO + '/blob/' + RAMO + '/' + SORGENTE + ')' : null,
      '',
      '### Cosa non va, o cosa migliorerei',
      '',
      '<!-- Scrivi qui: un errore, un passaggio poco chiaro, un esempio che non torna. -->',
      ''
      // Solo le righe condizionali valgono null: le stringhe vuote sono
      // separatori voluti, e in Markdown servono.
    ].filter(function (r) { return r !== null; }).join('\n');

    return REPO + '/issues/new?title=' + encodeURIComponent(titolo) +
           '&body=' + encodeURIComponent(corpo);
  }

  // Il libro è statico e non può autenticare nessuno: passa il testimone al
  // magazine, dove il lettore è loggato a casa sua. Il passaggio viaggia nel
  // querystring — niente CORS, niente cookie di terze parti.
  function urlLettura(citazione, sezione) {
    const dati = {
      q: tronca(citazione, MAX_CITAZIONE),
      p: window.location.origin + window.location.pathname +
         (sezione.ancora ? '#' + sezione.ancora : ''),
      s: sezione.titolo || '',
      f: SORGENTE || ''
    };
    const parti = Object.keys(dati)
      .filter(function (k) { return dati[k]; })
      .map(function (k) { return k + '=' + encodeURIComponent(dati[k]); });
    return LETTURA + (LETTURA.indexOf('?') === -1 ? '?' : '&') + parti.join('&');
  }

  // Puntatore grossolano = dita. È il segnale affidabile per distinguere
  // touch da mouse, molto più della larghezza dello schermo.
  const TOUCH = !!(window.matchMedia &&
                   window.matchMedia('(pointer: coarse)').matches);

  const ICONE = {
    // Fumetto: lo stesso del pulsante assistente sul magazine, così le due
    // superfici si riconoscono.
    ai: '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" ' +
        'stroke="currentColor" stroke-width="2" stroke-linecap="round" ' +
        'stroke-linejoin="round" aria-hidden="true" focusable="false">' +
        '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>' +
        '<path d="M9 9h.01M13 9h.01"/></svg>',
    // Matita: la segnalazione.
    segnala: '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" ' +
        'focusable="false"><path fill="currentColor" d="M11.013 1.427a1.75 1.75 0 0 1 ' +
        '2.474 0l1.086 1.086a1.75 1.75 0 0 1 0 2.474l-8.61 8.61c-.21.21-.47.364-.756.445' +
        'l-3.251.93a.75.75 0 0 1-.927-.928l.929-3.25c.081-.286.235-.547.445-.758l8.61-8.61Z' +
        'm1.414 1.06a.25.25 0 0 0-.354 0L10.811 3.75l1.439 1.44 1.263-1.263a.25.25 0 0 0 ' +
        '0-.354l-1.086-1.086Zm-1.238 3.763L9.75 4.81l-6.286 6.287a.253.253 0 0 0-.064.108' +
        'l-.558 1.953 1.953-.558a.253.253 0 0 0 .108-.064l6.286-6.286Z"/></svg>'
  };

  function segnalazioneDaSelezione() {
    // Le due voci sono indipendenti: ognuna compare solo se configurata.
    const voci = [
      { chiave: 'ai', attiva: !!LETTURA, etichetta: 'Chiedi all\u2019AI',
        titolo: 'Apri questo passaggio con l\u2019assistente di paithon.it' },
      { chiave: 'segnala', attiva: !!REPO, etichetta: 'Segnala',
        titolo: 'Apri una issue su GitHub sul testo selezionato' }
    ].filter(function (v) { return v.attiva; });
    if (!voci.length) return;

    const area = areaContenuto();
    if (!area) return;

    // Il gruppo fluttuante, creato una volta sola.
    const gruppo = document.createElement('div');
    gruppo.className = 'pt-azioni';
    gruppo.hidden = true;
    gruppo.setAttribute('role', 'group');
    gruppo.setAttribute('aria-label', 'Azioni sul testo selezionato');
    gruppo.innerHTML = voci.map(function (v) {
      return '<button type="button" class="pt-azioni__voce" data-azione="' + v.chiave +
             '" title="' + v.titolo + '">' + ICONE[v.chiave] +
             '<span>' + v.etichetta + '</span></button>';
    }).join('');
    document.body.appendChild(gruppo);

    let citazione = '';
    let sezione = { ancora: '', titolo: '' };

    function nascondi() {
      gruppo.hidden = true;
      gruppo.classList.remove('pt-azioni--visibile');
    }

    function posiziona(rect) {
      gruppo.hidden = false;

      if (TOUCH) {
        // Su touch il gruppo si ancora in basso al centro. Accanto alla
        // selezione non funzionerebbe: iOS e Android ci mettono il proprio
        // menu (Copia, Cerca, Condividi) a filo del testo scelto, e
        // coprirebbe qualunque cosa le si affianchi.
        gruppo.classList.add('pt-azioni--ancorata');
        gruppo.style.left = '';
        gruppo.style.top = '';
        gruppo.classList.add('pt-azioni--visibile');
        return;
      }

      // Misurabile solo da visibile: prima si mostra, poi si posiziona.
      const w = gruppo.offsetWidth;
      const h = gruppo.offsetHeight;
      const margine = 8;

      let x = rect.left + rect.width / 2 - w / 2;
      x = Math.max(margine, Math.min(x, window.innerWidth - w - margine));

      // Sotto la selezione; se non c'è spazio, sopra.
      let y = rect.bottom + margine;
      if (y + h > window.innerHeight - margine) y = rect.top - h - margine;
      y = Math.max(margine, y);

      gruppo.style.left = x + 'px';
      gruppo.style.top = y + 'px';

      // Sincrono, non in requestAnimationFrame: se la pagina non è "visibile"
      // (headless, scheda in secondo piano) i callback rAF non vengono
      // eseguiti e la pillola resterebbe per sempre a opacità zero. La
      // lettura di offsetWidth qui sopra ha già forzato il ricalcolo dello
      // stile, quindi la transizione parte comunque dallo stato nascosto.
      gruppo.classList.add('pt-azioni--visibile');
    }

    function valuta() {
      const sel = window.getSelection();
      if (!sel || sel.isCollapsed || sel.rangeCount === 0) return nascondi();

      const testo = sel.toString().trim().replace(/\s+/g, ' ');
      if (testo.length < MIN_CARATTERI) return nascondi();

      const range = sel.getRangeAt(0);
      // Solo testo del contenuto: non la sidebar, non l'indice, non il codice
      // dei blocchi copiabili — quelli si segnalano dal testo attorno.
      if (!area.contains(range.commonAncestorContainer)) return nascondi();

      const rect = range.getBoundingClientRect();
      if (!rect || (rect.width === 0 && rect.height === 0)) return nascondi();

      citazione = testo;
      sezione = sezioneDi(range.startContainer);
      posiziona(rect);
    }

    // `mousedown` con preventDefault: il click non deve azzerare la selezione
    // prima che si riesca a leggerla.
    gruppo.addEventListener('mousedown', function (e) { e.preventDefault(); });

    gruppo.addEventListener('click', function (e) {
      const voce = e.target.closest('[data-azione]');
      if (!voce || !citazione) return;
      const url = 'ai' === voce.dataset.azione
        ? urlLettura(citazione, sezione)
        : urlIssue(citazione, sezione);
      window.open(url, '_blank', 'noopener,noreferrer');
      nascondi();
    });

    let trascinando = false;
    let attesa = null;

    function programma(ritardo) {
      clearTimeout(attesa);
      attesa = setTimeout(valuta, ritardo);
    }

    // Il puntatore premuto azzera la selezione: si sparisce e si riparte dal
    // rilascio. Il guard sul gruppo evita di nascondersi prima del click.
    document.addEventListener('pointerdown', function (e) {
      if (gruppo.contains(e.target)) return;
      trascinando = true;
      nascondi();
    }, true);

    document.addEventListener('pointerup', function () { trascinando = false; programma(10); });
    document.addEventListener('mouseup', function () { trascinando = false; programma(10); });
    document.addEventListener('touchend', function () { trascinando = false; programma(140); });

    // Su touch `mouseup` NON esiste: si seleziona con le dita e con le
    // maniglie di sistema, e il solo evento che arriva è `selectionchange`.
    // Serve quindi anche come innesco per mostrare, non solo per nascondere —
    // ritardato, perché mentre si trascina ne arrivano decine di seguito.
    document.addEventListener('selectionchange', function () {
      const sel = window.getSelection();
      if (!sel || sel.isCollapsed) { nascondi(); return; }
      if (!trascinando) programma(180);
    });

    document.addEventListener('keyup', function (e) {
      const tasto = e.key || '';
      if (tasto === 'Shift' || tasto.indexOf('Arrow') === 0 ||
          (tasto === 'a' && (e.ctrlKey || e.metaKey))) programma(10);
    });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') nascondi();
    });

    // Scorrendo si RIPOSIZIONA, non si nasconde: su touch la selezione
    // sopravvive allo scorrimento, e sparire renderebbe il pulsante
    // irraggiungibile per un testo scelto poco sopra il bordo.
    window.addEventListener('scroll', function () {
      if (!gruppo.hidden) programma(60);
    }, { passive: true });
    window.addEventListener('resize', function () {
      if (!gruppo.hidden) programma(60);
    });
  }

  // ===== 3. LIVELLO DI LETTURA: ELEMENTARE O SUPERIORE =====
  // Il libro spiega ogni concetto su due livelli. Qui ci sono due decisioni
  // distinte, e vale la pena tenerle separate:
  //
  //   - il livello GLOBALE, scelto col pulsante in barra, da cui parte tutto
  //     il libro e che resta salvato;
  //   - l'apertura LOCALE di una singola tab, che non deve contagiare i
  //     blocchi successivi: aver letto in Superiore un passaggio non vuol dire
  //     volere il Superiore anche tre pagine dopo.
  const LIVELLI = ['elementare', 'superiore'];
  const CHIAVE_LIVELLO = 'pt-livello';

  function livelloCorrente() {
    const l = document.documentElement.dataset.ptLivello;
    return LIVELLI.indexOf(l) === -1 ? 'elementare' : l;
  }

  // Apre, dentro un blocco di tab, quella con l'etichetta del livello dato.
  // Restituisce false se il blocco non parla di livelli — le tab usate per
  // altro (varianti di codice, per esempio) vanno lasciate in pace.
  function apriLivello(insieme, livello) {
    const etichette = insieme.querySelectorAll(':scope > .tab-label');
    for (let i = 0; i < etichette.length; i++) {
      if (etichette[i].textContent.trim().toLowerCase() === livello) {
        const radio = document.getElementById(etichette[i].htmlFor);
        if (radio) radio.checked = true;
        return true;
      }
    }
    return false;
  }

  function applicaLivello(livello) {
    let toccati = 0;
    document.querySelectorAll('.tab-set').forEach(function (insieme) {
      if (apriLivello(insieme, livello)) toccati++;
    });
    return toccati;
  }

  // Solo aspetto: attributo su <html> (per l'icona) e parola sul pulsante.
  function mostraLivello(livello) {
    document.documentElement.dataset.ptLivello = livello;
    const nome = 'superiore' === livello ? 'Superiore' : 'Elementare';
    const testo = document.querySelector('[data-pt-livello-testo]');
    if (testo) testo.textContent = nome;
    const bottone = document.querySelector('[data-pt-livello-toggle]');
    if (bottone) {
      const altro = 'superiore' === livello ? 'Elementare' : 'Superiore';
      bottone.setAttribute('title', 'Livello ' + nome + ' — passa a ' + altro);
      bottone.setAttribute('aria-label',
        'Livello di lettura: ' + nome + '. Passa a ' + altro + '.');
      bottone.setAttribute('aria-pressed', String('superiore' === livello));
    }
  }

  // sphinx-inline-tabs assegna `label.onclick` per aprire, in TUTTA la pagina,
  // ogni tab con la stessa etichetta. È esattamente il contagio che non
  // vogliamo: si stacca. Il legame etichetta→radio è nativo dell'HTML, quindi
  // la singola tab continua ad aprirsi da sé.
  function scollegaTabSincronizzate() {
    document.querySelectorAll('.tab-label').forEach(function (et) {
      et.onclick = null;
    });
  }

  // `article_header_end` può solo aggiungere template in coda alla barra, e
  // il gruppo dei pulsanti del tema (chiaro/scuro, ricerca, indice di pagina)
  // è un blocco unico: da configurazione il nostro finisce prima di tutti —
  // appiccicato al logo — o dopo tutti. Nessuna delle due va bene. Quindi lo
  // spostiamo dentro quel gruppo, subito prima dell'indice di pagina:
  // penultimo, lontano dal logo, e l'ultima icona resta quella che i lettori
  // cercano lì. Se il tema cambiasse i nomi, si limita ad accodarsi.
  function collocaFraIPulsanti(bottone) {
    const gruppo = document.querySelector('.article-header-buttons');
    if (!gruppo || gruppo.contains(bottone)) return;

    const indicePagina = gruppo.querySelector('.secondary-toggle');
    if (indicePagina) {
      gruppo.insertBefore(bottone, indicePagina);
    } else {
      gruppo.appendChild(bottone);
    }

    // Il guscio del template resta vuoto: via, o lascia uno spazio fantasma.
    const guscio = document.querySelector('.pt-livello');
    if (guscio && !guscio.children.length) guscio.remove();
  }

  function livelloDiLettura() {
    let livello = null;

    // Un link condiviso con ?tabs=Superiore vince sulla preferenza salvata:
    // chi lo manda sta indicando un livello, e non deve cambiare il mio.
    const daUrl = new URLSearchParams(window.location.search).getAll('tabs')
      .map(function (s) { return s.trim().toLowerCase(); })
      .filter(function (s) { return LIVELLI.indexOf(s) !== -1; });
    if (daUrl.length) {
      livello = daUrl[0];
    } else {
      try {
        const salvato = localStorage.getItem(CHIAVE_LIVELLO);
        if (LIVELLI.indexOf(salvato) !== -1) livello = salvato;
      } catch (e) { /* niente localStorage */ }
    }

    mostraLivello(livello || livelloCorrente());

    // `setTimeout(…, 0)`: anche `ready()` di tabs.js gira su
    // DOMContentLoaded, e tutti i gestori sincroni finiscono prima del primo
    // timer — così non dipendiamo dall'ordine dei <script> nella pagina.
    setTimeout(function () {
      scollegaTabSincronizzate();
      if (livello) applicaLivello(livello);
    }, 0);

    const bottone = document.querySelector('[data-pt-livello-toggle]');
    if (!bottone) return;

    collocaFraIPulsanti(bottone);

    bottone.addEventListener('click', function () {
      const nuovo = 'superiore' === livelloCorrente() ? 'elementare' : 'superiore';
      mostraLivello(nuovo);
      try { localStorage.setItem(CHIAVE_LIVELLO, nuovo); } catch (e) {}
      applicaLivello(nuovo);
    });
  }

  // ===== AVVIO =====
  function avvia() {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', avvia);
      return;
    }
    leggiCoordinate();          // prima di tutto: i <meta> ora esistono
    statisticheRepository();
    segnalazioneDaSelezione();
    livelloDiLettura();
  }

  avvia();
})();
