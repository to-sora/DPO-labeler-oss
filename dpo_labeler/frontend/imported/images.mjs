import {$, element, model, canSave, message} from './common.mjs';

export function showImages(task) {
  $('images').replaceChildren();
  model.imagesReady = 0;
  if (!task.pair) return;
  task.pair.images.forEach((image, index) => {
    const card = element('article', undefined, 'card');
    card.append(element('h3', index === 0 ? 'Image A' : 'Image B'));
    const button = element('button', undefined, 'import-image-button');
    button.type = 'button';
    button.setAttribute('aria-label', `Enlarge image ${index === 0 ? 'A' : 'B'}`);
    const img = element('img');
    img.alt = `Image ${index === 0 ? 'A' : 'B'} — ${task.character_name}`;
    img.onload = () => {
      if (model.task !== task) return;
      model.imagesReady += 1;
      canSave();
    };
    img.onerror = () => message('Image unavailable. Check its source file, then refresh tasks.', true);
    img.src = `/media/imported/${encodeURIComponent(task.task_id)}/${image.image_id}`;
    button.append(img);
    button.onclick = () => {
      $('zoom').querySelector('img').src = img.src;
      $('zoom').showModal();
    };
    const prompt = element('details');
    prompt.append(element('summary', 'Prompt'), element('p', image.prompt || '(No prompt)'));
    card.append(button, prompt);
    $('images').append(card);
  });
}
