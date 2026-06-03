const debugCheckbox = document.getElementById('debugMode');
const saved = document.getElementById('saved');

function flashSaved() {
  saved.classList.add('show');
  setTimeout(() => saved.classList.remove('show'), 1200);
}

chrome.storage.local.get({ debugMode: false }, (settings) => {
  debugCheckbox.checked = !!settings.debugMode;
});

debugCheckbox.addEventListener('change', () => {
  chrome.storage.local.set({ debugMode: debugCheckbox.checked }, flashSaved);
});
