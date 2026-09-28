export const $ = (id) => document.getElementById(`import-${id}`);
export function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
export function message(text, error = false) {
  $('message').textContent = text;
  $('message').classList.toggle('error', error);
}
export async function request(path = '', body) {
  const response = await fetch(`/api/v1/imported/tasks${path}`, {
    method: body === undefined ? 'GET' : 'POST', cache: 'no-store',
    headers: {'Content-Type': 'application/json'},
    body: body === undefined ? undefined : JSON.stringify(body),
    signal: AbortSignal.timeout(120000),
  });
  if (!response.ok) {
    if (response.status === 401) document.dispatchEvent(new Event('dpo-labeler-auth-required'));
    const payload = await response.json();
    const error = new Error(payload.error || 'Request failed');
    error.status = response.status;
    throw error;
  }
  return response;
}
export async function data(path = '', body) {
  return (await (await request(path, body)).json()).data;
}
export const model = {tasks: [], task: null, choices: Object.create(null),
  busy: false, imagesReady: 0};
export function canSave() {
  const task = model.task;
  $('save').disabled = model.busy || !task?.pair || model.imagesReady !== 2 ||
    !$('reviewer').value.trim() || !task.dimensions.every(d => model.choices[d]);
}
