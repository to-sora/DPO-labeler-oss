const DEFECT_LABELS = {
  face_off: "Face off",
  eyes_off: "Eyes off",
  hand_corruption: "Corrupt hands",
  anatomy_other: "Other anatomy issue",
  bad_color_lighting: "Bad color or lighting",
  bad_composition: "Bad composition",
  bad_crop_framing: "Bad crop or framing",
  background_artifacts: "Background artifacts",
  low_detail_blur: "Low detail or blur",
  text_or_watermark_artifact: "Text or watermark artifact",
};

const DECISION_LABELS = {
  a_good: "A preferred",
  b_good: "B preferred",
  both_good: "Both Good",
  both_bad: "Both Bad",
  skip: "Skipped",
};

export function pairKey(datasetId, sessionId) {
  return `${datasetId}::${sessionId}`;
}

export function formatPromptText(value, fallback = "Not available") {
  const text = String(value ?? "").trim();
  return text || fallback;
}

export function formatDecision(value) {
  return DECISION_LABELS[value] || formatPromptText(value, "Not labeled");
}

export function formatDefectList(defects) {
  if (!Array.isArray(defects) || defects.length === 0) {
    return "None";
  }
  return defects.map((defect) => DEFECT_LABELS[defect] || defect).join(", ");
}

export function formatTimestamp(value) {
  const text = String(value ?? "").trim();
  if (!text) {
    return "Not available";
  }
  if (text.endsWith("Z")) {
    return `${text.slice(0, -1).replace("T", " ")} UTC`;
  }
  if (text.endsWith("+00:00")) {
    return `${text.slice(0, -6).replace("T", " ")} UTC`;
  }
  return text.replace("T", " ");
}

export function buildPairSummaryRows(pair) {
  return [];
}

export function buildLatestLabelRows(label) {
  if (!label) {
    return [summaryRow("Status", "Not labeled yet")];
  }
  return [
    summaryRow("Decision", formatDecision(label.decision)),
    summaryRow("Defects on A", formatDefectList(label.defects_a)),
    summaryRow("Defects on B", formatDefectList(label.defects_b)),
    summaryRow("Note", formatPromptText(label.note, "No note")),
    summaryRow("Updated", formatTimestamp(label.created_at)),
    summaryRow("Reviewer", label.reviewer_username),
  ];
}

export function buildImageSummaryRows(image) {
  return [
    summaryRow("Size", image.width && image.height ? `${image.width} x ${image.height}` : "Not available"),
    summaryRow("Steps / CFG", image.steps && image.cfg !== undefined && image.cfg !== null ? `${image.steps} / ${image.cfg}` : "Not available"),
  ];
}

export function buildCompactImageFacts(image) {
  const parts = [];
  if (image.width && image.height) {
    parts.push(`${image.width} x ${image.height}`);
  }
  if (image.steps) {
    parts.push(`${image.steps} steps`);
  }
  if (image.cfg !== undefined && image.cfg !== null && String(image.cfg).trim() !== "") {
    parts.push(`CFG ${image.cfg}`);
  }
  return parts.length ? parts.join(" · ") : "Details not available";
}

export function datasetProgress(dataset) {
  const reviewedPairs = (dataset?.tasks ?? []).reduce((sum, task) => sum + Number(task.reviewed_pairs || 0), 0);
  const totalPairs = (dataset?.tasks ?? []).reduce((sum, task) => sum + Number(task.total_pairs || 0), 0);
  const reviewedPercent = totalPairs ? Math.round((reviewedPairs / totalPairs) * 100) : 0;
  return {
    reviewedPairs,
    totalPairs,
    reviewedPercent,
  };
}

export function deriveTaskDisplayLabels(datasetDisplayName, task) {
  const normalizedDataset = String(datasetDisplayName || "").toLowerCase();
  const taskName = String(task?.task_name || "");
  const taskYamlName = String(task?.task_yaml_name || "");
  const taskYamlStem = taskYamlName.replace(/\.ya?ml$/i, "");
  return {
    title: containsNormalizedSubstring(normalizedDataset, taskYamlStem) ? "default" : taskYamlName,
    subtitle: containsNormalizedSubstring(normalizedDataset, taskName) ? "default" : taskName,
  };
}

export function applyOptimisticLabel(items, event) {
  return items.map((item) => {
    if (item.dataset_id !== event.dataset_id || item.session_id !== event.session_id) {
      return item;
    }
    return {
      ...item,
      is_labeled: true,
      latest_decision: event.decision,
      latest_reviewer_username: event.reviewer_username,
    };
  });
}

export function mergeQueueItems(existing, incoming) {
  const order = (existing ?? []).map((item) => pairKey(item.dataset_id, item.session_id));
  const byKey = new Map((existing ?? []).map((item) => [pairKey(item.dataset_id, item.session_id), { ...item }]));
  for (const item of incoming ?? []) {
    const key = pairKey(item.dataset_id, item.session_id);
    if (!byKey.has(key)) {
      order.push(key);
    }
    byKey.set(key, { ...byKey.get(key), ...item });
  }
  return order.map((key) => byKey.get(key));
}

export function deriveProgressSnapshot({ items, totalPairs }) {
  const queue = Array.isArray(items) ? items : [];
  const total = Number(totalPairs || 0);
  const labeledPairs = queue.filter((item) => item?.is_labeled).length;
  const firstUnlabeledIndex = queue.findIndex((item) => !item?.is_labeled);
  const nextIndex = firstUnlabeledIndex >= 0 ? firstUnlabeledIndex : queue.length;
  return {
    totalPairs: total,
    labeledPairs,
    firstUnlabeledIndex: nextIndex,
    queueComplete: total > 0 ? labeledPairs >= total : nextIndex >= queue.length,
  };
}

export function createKeyedValueCache({ maxWeight = Infinity, getWeight = () => 1 } = {}) {
  const entries = new Map();
  let clock = 0;

  function now() {
    clock += 1;
    return clock;
  }

  function totalWeight() {
    let sum = 0;
    for (const entry of entries.values()) {
      sum += entry.weight;
    }
    return sum;
  }

  function snapshot(key, entry) {
    return {
      key,
      value: entry.value,
      createdAt: entry.createdAt,
      lastAccessedAt: entry.lastAccessedAt,
      weight: entry.weight,
    };
  }

  return {
    has(key) {
      return entries.has(key);
    },
    get(key) {
      const entry = entries.get(key);
      if (!entry) {
        return null;
      }
      entry.lastAccessedAt = now();
      return entry.value;
    },
    peek(key) {
      return entries.get(key)?.value ?? null;
    },
    set(key, value) {
      const timestamp = now();
      const previous = entries.get(key);
      entries.set(key, {
        value,
        createdAt: previous?.createdAt ?? timestamp,
        lastAccessedAt: timestamp,
        weight: Math.max(0, Number(getWeight(value)) || 0),
      });
      return value;
    },
    touch(key) {
      const entry = entries.get(key);
      if (!entry) {
        return false;
      }
      entry.lastAccessedAt = now();
      return true;
    },
    delete(key) {
      const entry = entries.get(key);
      if (!entry) {
        return null;
      }
      entries.delete(key);
      return snapshot(key, entry);
    },
    clear() {
      const removed = [...entries.entries()].map(([key, entry]) => snapshot(key, entry));
      entries.clear();
      return removed;
    },
    entries() {
      return [...entries.entries()].map(([key, entry]) => snapshot(key, entry));
    },
    keys() {
      return [...entries.keys()];
    },
    totalWeight,
    evictToFit({ protectKeys = [] } = {}) {
      const protectedSet = new Set(protectKeys);
      const removed = [];
      while (totalWeight() > maxWeight) {
        const candidate = [...entries.entries()]
          .filter(([key]) => !protectedSet.has(key))
          .sort((left, right) => left[1].lastAccessedAt - right[1].lastAccessedAt)[0];
        if (!candidate) {
          break;
        }
        const [key, entry] = candidate;
        entries.delete(key);
        removed.push(snapshot(key, entry));
      }
      return removed;
    },
  };
}

export function randomizePairDisplay(pair, preferredDisplayOrder = null) {
  const images = sortImagesByIndex(pair.images);
  if (images.length !== 2) {
    return {
      ...pair,
      images,
      display_order: images.map((image) => Number(image.image_index)),
    };
  }
  const canonicalOrder = [Number(images[0].image_index), Number(images[1].image_index)];
  const fallbackDisplayOrder = Math.random() >= 0.5
    ? [Number(images[1].image_index), Number(images[0].image_index)]
    : canonicalOrder;
  const displayOrder = normalizeDisplayOrder(preferredDisplayOrder, fallbackDisplayOrder);
  const imageByIndex = new Map(images.map((image) => [Number(image.image_index), image]));
  return {
    ...pair,
    images: displayOrder.map((imageIndex) => imageByIndex.get(imageIndex)).filter(Boolean),
    display_order: displayOrder,
  };
}

export function mapLatestLabelToDisplay(label, currentDisplayOrder) {
  if (!label) {
    return null;
  }
  const savedDisplayOrder = normalizeDisplayOrder(label.display_order);
  const displayOrder = normalizeDisplayOrder(currentDisplayOrder, savedDisplayOrder);
  const chosenImageIndices = normalizeChosenImageIndices(label, savedDisplayOrder);
  const defectsByImageIndex = normalizeDefectsByImageIndex(label, savedDisplayOrder);
  return {
    ...label,
    decision: displayDecisionForCurrentOrder(label.decision, chosenImageIndices, displayOrder),
    defects_a: defectsByImageIndex[String(displayOrder[0])] ?? [],
    defects_b: defectsByImageIndex[String(displayOrder[1])] ?? [],
    display_order: displayOrder,
    chosen_image_indices: chosenImageIndices,
    defects_by_image_index: defectsByImageIndex,
  };
}

export function buildLabelPayloadFromDisplay(pair, decision, defectsA, defectsB) {
  const displayOrder = normalizeDisplayOrder(pair.display_order, pair.images.map((image) => Number(image.image_index)));
  const normalizedDefectsA = uniqueStrings(defectsA);
  const normalizedDefectsB = uniqueStrings(defectsB);
  return {
    decision,
    defects_a: normalizedDefectsA,
    defects_b: normalizedDefectsB,
    display_order: displayOrder,
    chosen_image_indices: deriveChosenImageIndices(decision, displayOrder),
    defects_by_image_index: {
      [String(displayOrder[0])]: normalizedDefectsA,
      [String(displayOrder[1])]: normalizedDefectsB,
    },
  };
}

export function buildEmptyFilter() {
  return {
    type: "group",
    operator: "and",
    conditions: [],
  };
}

export function cloneFilter(value) {
  return structuredClone(value);
}

export function summaryRow(label, value) {
  return {
    label,
    value: formatPromptText(value),
  };
}

function sortImagesByIndex(images) {
  return [...(images ?? [])].sort((left, right) => Number(left.image_index) - Number(right.image_index));
}

function normalizeDisplayOrder(order, fallback = [0, 1]) {
  if (Array.isArray(order) && order.length === 2) {
    const normalized = order.map((value) => Number(value));
    if (Number.isFinite(normalized[0]) && Number.isFinite(normalized[1]) && normalized[0] !== normalized[1]) {
      return normalized;
    }
  }
  return [...fallback];
}

function normalizeChosenImageIndices(label, savedDisplayOrder) {
  if (Array.isArray(label.chosen_image_indices)) {
    return uniqueNumbers(label.chosen_image_indices);
  }
  return deriveChosenImageIndices(label.decision, savedDisplayOrder);
}

function normalizeDefectsByImageIndex(label, savedDisplayOrder) {
  if (label?.defects_by_image_index && typeof label.defects_by_image_index === "object") {
    const normalized = {};
    for (const [imageIndex, defects] of Object.entries(label.defects_by_image_index)) {
      normalized[String(Number(imageIndex))] = uniqueStrings(defects);
    }
    return normalized;
  }
  return {
    [String(savedDisplayOrder[0])]: uniqueStrings(label.defects_a),
    [String(savedDisplayOrder[1])]: uniqueStrings(label.defects_b),
  };
}

function deriveChosenImageIndices(decision, displayOrder) {
  if (decision === "a_good") {
    return [displayOrder[0]];
  }
  if (decision === "b_good") {
    return [displayOrder[1]];
  }
  if (decision === "both_good") {
    return [...displayOrder].sort((left, right) => left - right);
  }
  return [];
}

function displayDecisionForCurrentOrder(originalDecision, chosenImageIndices, currentDisplayOrder) {
  if (originalDecision === "skip") {
    return "skip";
  }
  if (originalDecision === "both_bad") {
    return "both_bad";
  }
  if (chosenImageIndices.length === 2) {
    return "both_good";
  }
  if (chosenImageIndices.length === 1) {
    if (chosenImageIndices[0] === currentDisplayOrder[0]) {
      return "a_good";
    }
    if (chosenImageIndices[0] === currentDisplayOrder[1]) {
      return "b_good";
    }
  }
  return originalDecision;
}

function uniqueStrings(values) {
  if (!Array.isArray(values)) {
    return [];
  }
  return [...new Set(values.map((value) => String(value).trim()).filter(Boolean))];
}

function uniqueNumbers(values) {
  if (!Array.isArray(values)) {
    return [];
  }
  const normalized = [];
  for (const value of values) {
    const numeric = Number(value);
    if (Number.isFinite(numeric) && !normalized.includes(numeric)) {
      normalized.push(numeric);
    }
  }
  return normalized;
}

function containsNormalizedSubstring(normalizedHaystack, needle) {
  const normalizedNeedle = String(needle || "").toLowerCase();
  return Boolean(normalizedNeedle) && normalizedHaystack.includes(normalizedNeedle);
}
