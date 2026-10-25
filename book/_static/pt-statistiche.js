/*
 * Statistiche di visita del libro: quante persone leggono quali pagine.
 *
 * A ogni pagina parte una richiesta al Plausible di paithon.it, lo stesso
 * che conta le visite al magazine: niente cookie, niente identificativi, e
 * niente script di terzi, perche' il codice che gira e' tutto in questo file.
 * L'indirizzo che si manda e' quello canonico della pagina, cosi' le copie
 * del libro (la radice, /main/) contano come una pagina sola.
 *
 * Non parte fuori da book.paithon.it (in locale, in un'anteprima) ne' per
 * chi ha chiesto al browser di non essere tracciato (Global Privacy Control,
 * Do Not Track). Conta anche i download dei PDF, come evento a parte.
 */
(function () {
  'use strict';

  var DOMINIO = 'book.paithon.it';
  var API = 'https://paithon.it/api/event';

  if (location.hostname !== DOMINIO) { return; }
  if (navigator.globalPrivacyControl === true || navigator.doNotTrack === '1') { return; }

  function indirizzo() {
    var canonico = document.querySelector('link[rel="canonical"]');
    // Senza canonico, l'indirizzo senza query ne' hash: una ricerca nel
    // libro non deve uscire dalla pagina.
    return canonico && canonico.href ? canonico.href : location.origin + location.pathname;
  }

  function invia(nome, proprieta) {
    var dati = { n: nome, u: indirizzo(), d: DOMINIO, r: document.referrer || null };
    if (proprieta) { dati.p = proprieta; }
    try {
      // text/plain e no-cors: una richiesta semplice, senza preflight; la
      // risposta non serve.
      fetch(API, {
        method: 'POST',
        mode: 'no-cors',
        credentials: 'omit',
        keepalive: true,
        headers: { 'Content-Type': 'text/plain' },
        body: JSON.stringify(dati)
      });
    } catch (e) { /* un browser senza fetch non conta, e va bene cosi' */ }
  }

  function avvia() {
    invia('pageview');
    document.addEventListener('click', function (evento) {
      var link = evento.target && evento.target.closest ? evento.target.closest('a[href]') : null;
      if (link && /\.pdf(?:$|[?#])/i.test(link.href)) {
        invia('download', { url: link.href });
      }
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', avvia);
  } else {
    avvia();
  }
})();
