import {getSession} from '../app.js';
import {$, element, model, canSave} from './common.mjs';
import {refreshTasks, filterTasks, loadTask} from './catalog.mjs';
import {guarded, importTask, save, download} from './actions.mjs';

const response = await fetch('/imported/view.html', {cache: 'no-store'});
const template = document.createElement('template');
template.innerHTML = await response.text();
document.querySelector('.content').append(template.content);
const root = document.getElementById('imported-view');
const headline = document.querySelector('.topbar h1');
const originalHeadline = headline.textContent;
const button = element('button', 'External images', 'nav-btn');
button.id = 'nav-imported';
document.querySelector('.nav-actions').append(button);
const show = () => {
  document.body.classList.add('import-mode');
  root.hidden = false;
  $('manage').open = true;
  headline.textContent = 'External image review';
  button.textContent = 'Standard review';
  guarded(refreshTasks);
};
function leave() {
  document.body.classList.remove('import-mode');
  root.hidden = true;
  headline.textContent = originalHeadline;
  button.textContent = 'External images';
}
button.onclick = () => root.hidden ? show() : leave();
['nav-tasks', 'nav-review', 'nav-export', 'logout-button'].forEach(id => {
  document.getElementById(id).addEventListener('click', () => {
    leave();
  });
});
const entry = element('section', undefined, 'card panel import-entry');
entry.append(element('h2', 'Imported image tasks'));
const count = element('p', 'Import external images and grade each dimension.');
count.id = 'import-entry-count';
const open = element('button', 'Select imported tasks', 'nav-btn');
open.onclick = show;
entry.append(count, open);
document.getElementById('tasks-view').prepend(entry);
let signedIn = false;
function updateSession(session) {
  button.hidden = !session;
  if (!session) {
    signedIn = false;
    leave();
    model.task = null;
    model.tasks = [];
    $('work').hidden = true;
    $('images').replaceChildren();
    $('task-list').replaceChildren();
    $('task').replaceChildren(new Option('Select a task', ''));
    document.body.classList.remove('import-task-active');
    canSave();
  } else if (!signedIn) {
    signedIn = true;
    show();
  }
}
$('file').onchange = () => guarded(async () => {
  if ($('file').files[0]) $('yaml').value = await $('file').files[0].text();
});
$('create').onclick = () => guarded(importTask);
$('refresh').onclick = () => guarded(async () => {
  await refreshTasks(); await loadTask($('task').value);
});
$('filter').oninput = filterTasks;
$('task').onchange = () => guarded(() => loadTask($('task').value));
$('task-list').onclick = event => {
  const button = event.target.closest('button[data-task-id]');
  if (button) guarded(async () => {
    $('task').value = button.dataset.taskId;
    await loadTask(button.dataset.taskId);
  });
};
$('vote').onsubmit = event => { event.preventDefault(); guarded(save); };
$('export').onclick = () => guarded(download);
$('close').onclick = () => $('zoom').close();
document.addEventListener('dpo-labeler-session', event => updateSession(event.detail));
updateSession(getSession());
