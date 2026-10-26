/**
 * PAITHON BOOK: IL MENU COL DITO
 *
 * Sul telefono l'indice del libro sta in un pannello che il tema apre col
 * pulsante in alto a sinistra. Qui lo apre anche il dito: trascinando verso
 * destra il pannello entra dal bordo e resta attaccato al dito, e al rilascio
 * finisce di aprirsi o torna indietro, secondo quanto ne e' uscito e con che
 * slancio il dito l'ha lasciato. A pannello aperto, verso sinistra, lo stesso
 * al contrario.
 *
 * Il trascinamento che parte dal bordo sinistro dello schermo resta del
 * browser, che ci lega la pagina precedente (Safari, e Android con la
 * navigazione a gesti), e una pagina non ha modo di prenderselo: quel tocco
 * le arriva come `touchcancel`. Il gesto quindi non parte nella striscia
 * lungo il bordo (`BORDO`): li' il dito e' del browser, e il pannello non
 * comincia a uscire per poi tornare indietro quando il browser se lo prende.
 *
 * Il pannello lo apre e lo chiude il codice del tema, con un clic sul suo
 * pulsante e con `close()` sul suo <dialog>: focus, contenuto e chiusura col
 * tocco fuori restano quelli del tema. Lo script aggiunge lo spostamento
 * (`transform` e `--pt-gesto-menu`, vedi `pt-gesto-menu.css`). Il
 * `touchmove` non e' passivo perche', preso il pannello, deve tenere ferma
 * la pagina sotto; prima di allora non annulla niente.
 */

(function () {
  'use strict';

  // Sphinx emette due volte i file di `html_js_files`: senza la guardia un
  // trascinamento aprirebbe e richiuderebbe il pannello nello stesso tocco.
  if (window.__ptGestoMenuCaricato) return;
  window.__ptGestoMenuCaricato = true;

  // Quanti pixel CSS percorre il dito prima che il gesto si decida, e quanto
  // puo' deviare in verticale rispetto a quella corsa (0,5 vuol dire entro
  // 27 gradi dall'orizzontale: uno scorrimento un po' storto resta uno
  // scorrimento).
  const AVVIO = 10;
  const PENDENZA = 0.5;
  // Un dito rimasto fermo piu' a lungo sta premendo, non trascinando: la
  // pressione lunga seleziona il testo, o apre una postilla (450 ms in
  // `pt-postille.js`).
  const ATTESA = 400;
  // La striscia lungo il bordo sinistro in cui il dito e' del browser.
  const BORDO = 30;
  // Al rilascio decide lo slancio, se il dito correva almeno a SCATTO pixel
  // al millisecondo nell'ultimo decimo di secondo (FINESTRA) e aveva fatto
  // almeno SCATTO_MINIMO pixel; altrimenti decide quanto pannello si vede.
  const SCATTO = 0.4;
  const SCATTO_MINIMO = 30;
  const FINESTRA = 100;
  // Da questa larghezza in su l'indice sta accanto al testo e il pulsante
  // del tema lo nasconde invece di aprirlo (`fixSidebarToggle` di
  // sphinx-book-theme): li' il gesto non fa niente.
  const LARGO = '(min-width: 992px)';

  let tocco = null;   // il dito appoggiato, finche' il gesto non si decide
  let traino = null;  // il pannello attaccato al dito
  let posa = null;    // il pannello che va al suo posto dopo il rilascio
  let chiusoDalDito = false;  // il gesto ha chiuso il <dialog>

  function pannello() {
    return document.getElementById('pst-primary-sidebar-modal');
  }

  // Il pulsante c'e' anche dove il tema lo nasconde col CSS: un pulsante
  // senza riquadri non e' sullo schermo, e il gesto non apre niente.
  function pulsanteVisibile() {
    const b = document.querySelector('.primary-toggle');
    return b && b.getClientRects().length ? b : null;
  }

  // Un contenitore che scorre di lato (un blocco di codice, una tabella, una
  // formula larga) si tiene il dito.
  function scorreDiLato(el) {
    for (; el && el.nodeType === 1 && el !== document.body;
         el = el.parentElement) {
      if (el.scrollWidth > el.clientWidth + 1) {
        const ox = getComputedStyle(el).overflowX;
        if (ox === 'auto' || ox === 'scroll') return true;
      }
    }
    return false;
  }

  function selezioneAperta() {
    const s = window.getSelection && window.getSelection();
    return !!s && !s.isCollapsed;
  }

  // Un altro <dialog> aperto (la ricerca, la lente delle figure) ha il dito
  // per se'.
  function altroDialogo(menu) {
    return Array.prototype.some.call(
      document.querySelectorAll('dialog[open]'),
      function (d) { return d !== menu; });
  }

  // Con la pagina ingrandita il trascinamento di lato sposta la vista.
  function ingrandita() {
    return !!window.visualViewport && window.visualViewport.scale > 1.05;
  }

  function puoPartire(evento, menu) {
    if (evento.touches.length !== 1) return false;
    if (window.matchMedia(LARGO).matches) return false;
    if (!menu || altroDialogo(menu) || ingrandita() || selezioneAperta()) {
      return false;
    }
    const bersaglio = evento.target;
    if (bersaglio.closest && bersaglio.closest(
      'input, textarea, select, [contenteditable], video, audio')) {
      return false;
    }
    if (menu.open) return true;
    if (evento.touches[0].clientX < BORDO) return false;
    return !!pulsanteVisibile() && !scorreDiLato(bersaglio);
  }

  // Il dito ha fatto AVVIO pixel: il pannello e' suo se va di lato e nel
  // verso giusto, se si e' mosso subito, a pagina ferma e senza selezione.
  // I tempi sono quelli degli eventi, cioe' di quando il dito si e' mosso, e
  // non di quando lo script li legge: se la pagina e' occupata (aprire il
  // pannello sposta tutto l'indice), il dito non sembra piu' lento di com'e'.
  function prende(da, dx, dy, menu, quando) {
    if (dy > Math.abs(dx) * PENDENZA) return false;
    if (quando - da.quando > ATTESA) return false;
    if (Math.abs(window.scrollY - da.scorrimento) > AVVIO) return false;
    if (selezioneAperta()) return false;
    if (!menu || menu.open !== da.aperto) return false;
    return menu.open ? dx < 0 : dx > 0;
  }

  // `pos` e' lo spostamento del pannello: 0 aperto, -larghezza fuori.
  function sposta(menu, pos, larghezza) {
    menu.style.transform = 'translateX(' + pos + 'px)';
    menu.style.setProperty('--pt-gesto-menu', String(1 + pos / larghezza));
  }

  function pulisci(menu) {
    menu.classList.remove('pt-gesto-segue', 'pt-gesto-posa');
    menu.style.removeProperty('transform');
    menu.style.removeProperty('--pt-gesto-menu');
  }

  // Alla chiusura il tema restituisce le classi del pannello all'elemento
  // della pagina (`#pst-primary-sidebar`), che ha la transizione del tema sul
  // margine: l'indice riapparirebbe aperto e scivolerebbe via una seconda
  // volta. Chiuso col tocco fuori o col pulsante, quella scivolata fa da
  // uscita; chiuso col dito, il pannello e' gia' fuori, e lo si mette a posto
  // senza transizione. Il gestore del tema e' registrato all'avvio della
  // pagina e questo dopo, al primo gesto: le classi sono gia' tornate.
  function fermaSidebar() {
    const s = document.getElementById('pst-primary-sidebar');
    if (!s) return;
    s.style.transition = 'none';
    void getComputedStyle(s).marginLeft;
    s.style.removeProperty('transition');
  }

  // Se il pannello si chiude per un'altra via (Esc, una voce dell'indice)
  // mentre il dito lo tiene o mentre va a posto, lo spostamento si toglie.
  function sorveglia(menu) {
    if (menu.__ptGestoSorvegliato) return;
    menu.__ptGestoSorvegliato = true;
    menu.addEventListener('close', function () {
      if (traino && traino.menu === menu) traino = null;
      if (posa && posa.menu === menu) {
        clearTimeout(posa.timer);
        posa = null;
      }
      pulisci(menu);
      if (chiusoDalDito) {
        chiusoDalDito = false;
        fermaSidebar();
      }
    });
  }

  function segui(t, quando) {
    const dx = t.clientX - traino.x;
    const pos = traino.aperto ? dx : dx - traino.larghezza;
    traino.pos = Math.min(0, Math.max(-traino.larghezza, pos));
    sposta(traino.menu, traino.pos, traino.larghezza);
    // Solo i campioni dell'ultimo decimo di secondo: uno piu' vecchio
    // porterebbe nel conto una pausa che non c'entra con lo slancio, e un
    // dito che si ferma prima di staccarsi ne lascia uno solo, cioe' zero.
    const c = traino.campioni;
    c.push({ t: quando, x: t.clientX });
    while (quando - c[0].t > FINESTRA) c.shift();
  }

  function slancio() {
    const c = traino.campioni;
    const tempo = c[c.length - 1].t - c[0].t;
    return tempo > 0 ? (c[c.length - 1].x - c[0].x) / tempo : 0;
  }

  // Il pannello e' arrivato: se doveva chiudersi lo chiude il <dialog>, che
  // riporta l'indice nella pagina.
  function concludiPosa() {
    if (!posa) return;
    const p = posa;
    posa = null;
    clearTimeout(p.timer);
    if (!p.aperto && p.menu.open) {
      chiusoDalDito = true;
      p.menu.close();
    }
    pulisci(p.menu);
  }

  function rilascia(aperto) {
    const t = traino;
    traino = null;
    const menu = t.menu;
    // La transizione parte dal punto in cui il dito ha lasciato il pannello:
    // lo si fissa prima di cambiarla.
    void getComputedStyle(menu).transform;
    menu.classList.replace('pt-gesto-segue', 'pt-gesto-posa');
    sposta(menu, aperto ? 0 : -t.larghezza, t.larghezza);
    posa = { menu: menu, aperto: aperto, timer: 0 };
    // La durata e' quella del brand, che vale zero col movimento ridotto.
    const durata =
      parseFloat(getComputedStyle(menu).transitionDuration) * 1000 || 0;
    if (durata > 0) posa.timer = setTimeout(concludiPosa, durata + 30);
    else concludiPosa();
  }

  document.addEventListener('touchstart', function (evento) {
    tocco = null;
    // Un dito che arriva mentre il pannello va a posto lo trova arrivato; un
    // secondo dito rimanda il pannello dove l'aveva trovato il primo.
    concludiPosa();
    if (traino) rilascia(traino.aperto);
    const menu = pannello();
    if (!puoPartire(evento, menu)) return;
    const t = evento.touches[0];
    tocco = {
      x: t.clientX, y: t.clientY, quando: evento.timeStamp,
      scorrimento: window.scrollY, aperto: menu.open,
    };
  }, { passive: true });

  document.addEventListener('touchmove', function (evento) {
    if (traino) {
      if (evento.touches.length !== 1) { rilascia(traino.aperto); return; }
      if (evento.cancelable) evento.preventDefault();
      segui(evento.touches[0], evento.timeStamp);
      return;
    }
    if (!tocco) return;
    if (evento.touches.length !== 1) { tocco = null; return; }
    const t = evento.touches[0];
    const dx = t.clientX - tocco.x;
    const dy = Math.abs(t.clientY - tocco.y);
    if (Math.abs(dx) < AVVIO && dy < AVVIO) return;
    const da = tocco;
    tocco = null;
    const menu = pannello();
    if (!prende(da, dx, dy, menu, evento.timeStamp)) return;
    if (!menu.open) {
      const pulsante = pulsanteVisibile();
      if (pulsante) pulsante.click();
      if (!menu.open) return;
    }
    if (evento.cancelable) evento.preventDefault();
    sorveglia(menu);
    traino = {
      menu: menu, x: da.x, aperto: da.aperto,
      larghezza: menu.offsetWidth || 1, pos: 0, campioni: [],
    };
    // Il tema ha appena aperto il <dialog>: il pannello va spostato prima
    // che il browser lo disegni aperto del tutto.
    menu.classList.add('pt-gesto-segue');
    segui(t, evento.timeStamp);
  }, { passive: false });

  document.addEventListener('touchend', function (evento) {
    tocco = null;
    if (!traino || evento.touches.length !== 0) return;
    const t = evento.changedTouches[0];
    segui(t, evento.timeStamp);
    const v = slancio();
    let aperto = traino.pos > -traino.larghezza / 2;
    if (Math.abs(v) >= SCATTO &&
        Math.abs(t.clientX - traino.x) >= SCATTO_MINIMO) {
      aperto = v > 0;
    }
    rilascia(aperto);
  }, { passive: true });

  // Il browser si e' preso il tocco (il gesto di sistema, uno scorrimento):
  // il pannello torna com'era.
  document.addEventListener('touchcancel', function () {
    tocco = null;
    if (traino) rilascia(traino.aperto);
  }, { passive: true });
})();
