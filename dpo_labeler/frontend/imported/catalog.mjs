import {$, data, element, message, model, canSave} from './common.mjs';
import {render} from './render.mjs';

let loadNumber = 0;
export const pendingKey = id => `imported-pending-${id}`;

export function filterTasks() {
  const current = $('task').value;
  const filter = $('filter').value.toLocaleLowerCase();
  const tasks = model.tasks.filter(t =>
    `${t.task_name} ${t.character_name} ${t.image_dir}`.toLocaleLowerCase().includes(filter));
  $('task').replaceChildren(new Option(tasks.length ? 'Select a task' : 'No matching tasks', ''));
  tasks.forEach(t => $('task').add(new Option(
    `${t.task_name} · ${t.character_name} (${t.image_count} images)`, t.task_id)));
  if (tasks.some(t => t.task_id === current)) $('task').value = current;
}

export async function refreshTasks() {
  const result = await data();
  model.tasks = result.tasks;
  filterTasks();
  if (result.warnings.length) message(result.warnings.join('\n'), true);
  const count = document.getElementById('import-entry-count');
  if (count) count.textContent = `${result.tasks.length} imported tasks`;
}

export async function loadTask(id) {
  const number = ++loadNumber;
  model.task = null;
  document.body.classList.remove('import-task-active');
  $('work').hidden = true;
  canSave();
  if (!id) return;
  message('Loading task…');
  const pending = localStorage.getItem(pendingKey(id));
  if (pending) {
    try {
      await data(`/${encodeURIComponent(id)}/comparisons`, JSON.parse(pending));
      localStorage.removeItem(pendingKey(id));
    } catch (error) {
      if (error.status !== 409) throw error;
      localStorage.removeItem(pendingKey(id));
      message(error.message, true);
    }
  }
  const task = await data(`/${encodeURIComponent(id)}`);
  if (number !== loadNumber) return;
  render(task);
  $('save').textContent = 'Save all dimensions & continue';
  message(task.complete ? 'All dimensions are certified.' : 'Choose a preference in each open dimension.');
}
