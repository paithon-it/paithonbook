/**
 * PAITHON BOOK: IL MENU COL DITO
 *
 * Sul telefono l'indice del libro sta in un pannello che il tema apre col
 * pulsante in alto a sinistra. Qui lo apre anche il dito: un trascinamento
 * verso destra lo fa entrare, uno verso sinistra lo richiude.
 *
 * Il trascinamento che parte dal bordo sinistro dello schermo resta del
 * browser, che ci lega la pagina precedente (Safari, e Android con la
 * navigazione a gesti), e una pagina non ha modo di prenderselo: quel tocco
 * le arriva come `touchcancel`. Il gesto che apre il pannello comincia quindi
 * dentro la pagina.
 *
 * Il pannello lo apre e lo chiude il codice del tema, con un clic sul suo
 * pulsante e con `close()` sul suo <dialog>: focus, contenuto e chiusura col
 * tocco fuori restano quelli del tema. I gestori sono passivi e decidono
 * solo al rilascio, quindi lo scorrimento della pagina non aspetta mai.
 */

(function () {
  'use strict';

  // Sphinx emette due volte i file di `html_js_files`: senza la guardia un
  // trascinamento aprirebbe e richiuderebbe il pannello nello stesso tocco.
  if (window.__ptGestoMenuCaricato) return;
  window.__ptGestoMenuCaricato = true;

  // Quanto deve correre il dito in orizzontale (in pixel CSS), quanto puo'
  // deviare in verticale rispetto alla corsa (0,5 vuol dire entro 27 gradi
  // dall'orizzontale: uno scorrimento verticale un po' storto non conta) e
  // quanto puo' durare (oltre, il dito sta leggendo o selezionando).
  const CORSA = 60;
  const PENDENZA = 0.5;
  const DURATA = 700;
  // Da questa larghezza in su l'indice sta accanto al testo e il pulsante
  // del tema lo nasconde invece di aprirlo (`fixSidebarToggle` di
  // sphinx-book-theme): li' il gesto non fa niente.
  const LARGO = '(min-width: 992px)';

  let inizio = null;

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
    return !!pulsanteVisibile() && !scorreDiLato(bersaglio);
  }

  document.addEventListener('touchstart', function (evento) {
    // Un secondo dito che si appoggia annulla il gesto del primo.
    inizio = null;
    const menu = pannello();
    if (!puoPartire(evento, menu)) return;
    const t = evento.touches[0];
    inizio = {
      x: t.clientX, y: t.clientY, deviazione: 0,
      quando: performance.now(), scorrimento: window.scrollY,
      aperto: menu.open,
    };
  }, { passive: true });

  document.addEventListener('touchmove', function (evento) {
    if (!inizio) return;
    if (evento.touches.length !== 1) { inizio = null; return; }
    const dy = Math.abs(evento.touches[0].clientY - inizio.y);
    if (dy > inizio.deviazione) inizio.deviazione = dy;
  }, { passive: true });

  document.addEventListener('touchcancel', function () {
    inizio = null;
  }, { passive: true });

  document.addEventListener('touchend', function (evento) {
    const da = inizio;
    inizio = null;
    if (!da || evento.touches.length !== 0) return;
    const t = evento.changedTouches[0];
    const dx = t.clientX - da.x;
    const dy = Math.max(Math.abs(t.clientY - da.y), da.deviazione);
    if (Math.abs(dx) < CORSA || dy > Math.abs(dx) * PENDENZA) return;
    if (performance.now() - da.quando > DURATA) return;
    // La pagina si e' mossa: era uno scorrimento, non un gesto.
    if (Math.abs(window.scrollY - da.scorrimento) > 10) return;
    if (selezioneAperta()) return;
    const menu = pannello();
    if (!menu || menu.open !== da.aperto) return;
    if (dx > 0 && !menu.open) {
      const pulsante = pulsanteVisibile();
      if (pulsante) pulsante.click();
    } else if (dx < 0 && menu.open) {
      menu.close();
    }
  }, { passive: true });
})();
