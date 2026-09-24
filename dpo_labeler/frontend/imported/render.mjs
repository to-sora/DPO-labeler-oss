import {$, element, model, canSave} from './common.mjs';
import {showImages} from './images.mjs';

export function render(task) {
  model.task = task;
  document.body.classList.add('import-task-active');
  $('manage').open = false;
  model.choices = Object.assign(Object.create(null), task.pair?.locked || {});
  $('work').hidden = false;
  $('title').textContent = `${task.task_name} · ${task.character_name}`;
  $('source').textContent = task.image_dir;
  $('output').textContent = `Task cache and exports: ${task.output_dir}`;
  const completed = task.rankings.filter(r => r.certified).length;
  $('progress').textContent = `${task.images.length} images · ${task.comparisons} comparisons · ` +
    `${completed}/${task.dimensions.length} dimensions certified · Max exposure ${Math.max(...task.exposure)}`;
  $('dimension').replaceChildren(new Option('All dimensions', ''));
  task.dimensions.forEach((d, i) => $('dimension').add(new Option(d, String(i))));
  $('complete').hidden = !task.complete;
  $('vote').hidden = !task.pair;
  $('choices').replaceChildren();
  showImages(task);
  if (task.pair) task.dimensions.forEach((dim, i) => {
    const group = element('fieldset', undefined, 'import-choice');
    group.append(element('legend', dim));
    const locked = Object.hasOwn(task.pair.locked, dim) ? task.pair.locked[dim] : null;
    group.disabled = Boolean(locked);
    ['a_good', 'b_good'].forEach((choice, side) => {
      const label = element('label');
      const input = element('input');
      input.type = 'radio';
      input.name = `dimension-${i}`;
      input.value = choice;
      input.required = true;
      input.checked = model.choices[dim] === choice;
      input.onchange = () => {model.choices[dim] = choice; canSave();};
      label.append(input, element('span', side === 0 ? 'A' : 'B'));
      group.append(label);
    });
    if (locked) group.append(element('span', 'Inferred from saved comparisons'));
    $('choices').append(group);
  });
  canSave();
}
