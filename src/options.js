const debugCheckbox = document.getElementById('debugMode');
const fastCheckbox = document.getElementById('fastDebug');
const fastLabel = document.getElementById('fastDebugLabel');
const saved = document.getElementById('saved');

function flashSaved() {
  saved.classList.add('show');
  setTimeout(() => saved.classList.remove('show'), 1200);
}

// Fast debug only makes sense while debug mode is on — grey it out otherwise.
function syncFastEnabled() {
  const on = debugCheckbox.checked;
  fastCheckbox.disabled = !on;
  fastLabel.style.opacity = on ? '1' : '0.45';
  fastLabel.style.cursor = on ? 'pointer' : 'not-allowed';
}

chrome.storage.local.get({ debugMode: false, fastDebug: false }, (settings) => {
  debugCheckbox.checked = !!settings.debugMode;
  fastCheckbox.checked = !!settings.fastDebug;
  syncFastEnabled();
});

debugCheckbox.addEventListener('change', () => {
  chrome.storage.local.set({ debugMode: debugCheckbox.checked }, flashSaved);
  syncFastEnabled();
});

fastCheckbox.addEventListener('change', () => {
  chrome.storage.local.set({ fastDebug: fastCheckbox.checked }, flashSaved);
});
