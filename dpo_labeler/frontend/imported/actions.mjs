import {$, data, request, model, message, canSave} from './common.mjs';
import {pendingKey, refreshTasks, loadTask} from './catalog.mjs';

export async function guarded(action) {
  if (model.busy) return;
  model.busy = true;
  const ids = ['create', 'refresh', 'task', 'file', 'export'];
  ids.forEach(id => $(id).disabled = true);
  canSave();
  try { await action(); }
  catch (error) { message(error.message, true); }
  finally {
    model.busy = false;
    ids.forEach(id => $(id).disabled = false);
    canSave();
  }
}

export async function importTask() {
  message('Checking source images and importing…');
  const task = await data('', {yaml: $('yaml').value});
  $('filter').value = '';
  await refreshTasks();
  $('task').value = task.task_id;
  await loadTask(task.task_id);
  $('yaml').closest('details').open = false;
}

export async function save() {
  const id = model.task.task_id;
  let client = localStorage.getItem('imported-client');
  if (!client) {
    client = crypto.randomUUID();
    localStorage.setItem('imported-client', client);
  }
  const key = pendingKey(id);
  const payload = JSON.parse(localStorage.getItem(key) || 'null') || {
    comparison_id: model.task.pair.comparison_id,
    reviewer_username: $('reviewer').value.trim(), client_instance_id: client,
    choices: model.choices,
  };
  localStorage.setItem(key, JSON.stringify(payload));
  $('choices').querySelectorAll('fieldset').forEach(f => f.disabled = true);
  try {
    await data(`/${encodeURIComponent(id)}/comparisons`, payload);
    localStorage.removeItem(key);
  } catch (error) {
    if (error.status === 409) {
      localStorage.removeItem(key);
      await loadTask(id);
    } else $('save').textContent = 'Retry saved comparison';
    throw error;
  }
  await refreshTasks();
  await loadTask(id);
}

export async function download() {
  const selected = $('dimension').value;
  const response = await request(`/${encodeURIComponent(model.task.task_id)}/export`,
    {dimension: selected === '' ? null : Number(selected)});
  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement('a');
  link.href = url;
  link.download = selected === '' ? 'all-dimensions.zip' : `dimension-${selected}.zip`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
  message('Export saved in this task’s output folder and downloaded.');
}
