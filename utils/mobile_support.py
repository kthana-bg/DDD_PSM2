import json

import streamlit as st
from utils.html_embed import embed

_HOST_JS = """
var host = window;
try { if (window.parent && window.parent.document) { host = window.parent; } } catch (e) { host = window; }
"""


def _js_string(value):
    """Safely embed a Python string inside a JavaScript snippet."""
    return json.dumps(str(value)).replace("</", "<\\/")


# ----------------------------------------------------------------------------
# HTML builders
# ----------------------------------------------------------------------------
def build_head_tags_html():
    return """
    <script>
    (function() {
      """ + _HOST_JS + """
      try {
        var d = host.document;
        function setMeta(name, content) {
          var m = d.querySelector('meta[name="' + name + '"]');
          if (!m) { m = d.createElement('meta'); m.setAttribute('name', name); d.head.appendChild(m); }
          m.setAttribute('content', content);
        }
        setMeta('theme-color', '#0f1117');
        setMeta('mobile-web-app-capable', 'yes');
        setMeta('apple-mobile-web-app-capable', 'yes');
        setMeta('apple-mobile-web-app-status-bar-style', 'black-translucent');
        setMeta('apple-mobile-web-app-title', 'DrowsyGuard');
      } catch (e) { console.log('DrowsyGuard head tags:', e); }
    })();
    </script>
    """


def build_keep_awake_html(active):
    return """
    <div id="s" style="font: 12px/1.4 sans-serif; color: #aaa; padding: 2px 0;"></div>
    <script>
    (function() {
      """ + _HOST_JS + """
      var want = __ACTIVE__;
      var statusEl = document.getElementById('s');
      function show(text, color) { statusEl.textContent = text; statusEl.style.color = color; }

      if (!host.navigator || !('wakeLock' in host.navigator)) {
        show(want ? 'This browser cannot keep the screen on. Turn off auto-lock while monitoring.' : '', '#f39c12');
        return;
      }

      async function acquire() {
        try {
          if (host.__dgWake && !host.__dgWake.released) {
            show('Screen stays on while monitoring', '#2ecc71');
            return;
          }
          host.__dgWake = await host.navigator.wakeLock.request('screen');
          show('Screen stays on while monitoring', '#2ecc71');
        } catch (e) {
          if (e.name !== 'NotAllowedError') show('Could not keep the screen on (' + e.name + ')', '#f39c12');
        }
      }

      async function release() {
        try { if (host.__dgWake) { await host.__dgWake.release(); } } catch (e) {}
        host.__dgWake = null;
        show('', '#aaa');
      }

      host.__dgWantWake = want;
      if (want) {
        acquire();
        // the browser drops the lock when the tab is hidden; take it again when we come back
        if (!host.__dgWakeListener) {
          host.__dgWakeListener = true;
          host.document.addEventListener('visibilitychange', function() {
            if (host.document.visibilityState === 'visible' && host.__dgWantWake) { acquire(); }
          });
        }
      } else {
        release();
      }
    })();
    </script>
    """.replace("__ACTIVE__", "true" if active else "false")


def build_notify_html(title, body, vibrate=True):
    return """
    <script>
    (function() {
      """ + _HOST_JS + """
      try {
        var title = __TITLE__;
        var body = __BODY__;
        if (__VIBRATE__ && host.navigator && host.navigator.vibrate) {
          try { host.navigator.vibrate([200, 100, 200]); } catch (e) {}
        }
        if (host.document.hidden) {
          if (!host.__dgOrigTitle) { host.__dgOrigTitle = host.document.title; }
          host.document.title = '\\u26A0 ' + title;
          if (!host.__dgTitleListener) {
            host.__dgTitleListener = true;
            host.document.addEventListener('visibilitychange', function() {
              if (!host.document.hidden && host.__dgOrigTitle) { host.document.title = host.__dgOrigTitle; }
            });
          }
          if (host.Notification && host.Notification.permission === 'granted') {
            try { new host.Notification(title, { body: body, tag: 'drowsyguard-alert', renotify: true }); }
            catch (e) { /* Android Chrome only allows notifications from a service worker */ }
          }
        }
      } catch (e) { console.log('DrowsyGuard alert:', e); }
    })();
    </script>
    """.replace("__TITLE__", _js_string(title)).replace("__BODY__", _js_string(body)) \
       .replace("__VIBRATE__", "true" if vibrate else "false")


def build_notification_button_html():
    return """
    <div style="font: 13px/1.4 sans-serif; color: #ccc;">
      <button id="b" style="background:#3498db;color:#fff;border:0;border-radius:6px;padding:10px 14px;font-size:13px;cursor:pointer;">
        Enable alert notifications
      </button>
      <span id="s" style="margin-left:8px;"></span>
    </div>
    <script>
    (function() {
      """ + _HOST_JS + """
      var btn = document.getElementById('b');
      var st = document.getElementById('s');
      var N = host.Notification;
      function paint() {
        if (!N) {
          btn.style.display = 'none';
          st.textContent = 'Notifications are not supported on this device. You will still get sound, voice and vibration.';
          return;
        }
        if (N.permission === 'granted') { btn.style.display = 'none'; st.textContent = 'Notifications enabled'; st.style.color = '#2ecc71'; }
        else if (N.permission === 'denied') { btn.style.display = 'none'; st.textContent = 'Notifications are blocked in the browser settings.'; st.style.color = '#f39c12'; }
        else { btn.style.display = 'inline-block'; st.textContent = ''; }
      }
      btn.addEventListener('click', function() {
        try {
          var r = N.requestPermission(function() { paint(); });
          if (r && r.then) { r.then(paint); }
        } catch (e) { paint(); }
      });
      paint();
    })();
    </script>
    """


# ----------------------------------------------------------------------------
# Streamlit-facing API
# ----------------------------------------------------------------------------
def inject_head_tags():
    """Once per session: add mobile/home-screen metadata to the page head."""
    if st.session_state.get("_dg_head_tags_done"):
        return
    st.session_state["_dg_head_tags_done"] = True
    embed(build_head_tags_html(), height=0)


def keep_awake(active):
    """Keep the screen on while a session is running (and release the lock when it stops)."""
    embed(build_keep_awake_html(bool(active)), height=30 if active else 0)


def notify_alert(title, body, vibrate=True):
    """Vibrate, flag the tab title and (if allowed) pop a notification when the tab is hidden."""
    embed(build_notify_html(title, body, vibrate), height=0)


def render_notification_button():
    embed(build_notification_button_html(), height=48)
