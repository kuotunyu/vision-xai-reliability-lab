"use strict";

const models = {
  cnn: {
    label: "CNN",
    variant: "cnn",
    localization: "assets/figures/localization_cnn.png",
    faithfulness: "assets/figures/faithfulness_cnn.png",
    spurious: "assets/figures/spurious_cnn_patched.png",
    spuriousVariant: "cnn_patched",
    spuriousMethod: "gradcam",
  },
  vit: {
    label: "ViT",
    variant: "vit",
    localization: "assets/figures/localization_vit.png",
    faithfulness: "assets/figures/faithfulness_vit.png",
    spurious: "assets/figures/spurious_vit_patched.png",
    spuriousVariant: "vit_patched",
    spuriousMethod: "integrated_gradients",
  },
};

const baselineMethods = new Set(["center", "random", "uniform"]);
let fullSummary = null;

function mean(payload) {
  return Number(payload.mean);
}

function bestAttributionPointing(summary, variant) {
  return Math.max(
    ...Object.entries(summary.localization[variant])
      .filter(([method]) => !baselineMethods.has(method))
      .map(([, payload]) => mean(payload.all.pointing_rate)),
  );
}

function spuriousSpread(summary, model) {
  const tests = summary.spurious[model.spuriousVariant][model.spuriousMethod];
  const values = Object.values(tests).map((payload) => mean(payload.accuracy));
  return Math.max(...values) - Math.min(...values);
}

function setMetric(name, value) {
  document.querySelectorAll("[data-metric='" + name + "']").forEach((element) => {
    element.textContent = value;
  });
}

function selectModel(key) {
  const model = models[key];
  document.querySelectorAll("[data-model]").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.model === key));
  });

  const localization = document.querySelector("#localization-figure");
  const faithfulness = document.querySelector("#faithfulness-figure");
  const spurious = document.querySelector("#spurious-figure");
  localization.src = model.localization;
  localization.alt =
    model.label + " localization aggregate chart with 95 percent bootstrap confidence intervals";
  faithfulness.src = model.faithfulness;
  faithfulness.alt = model.label + " deletion and insertion faithfulness curves";
  spurious.src = model.spurious;
  spurious.alt = model.label + " spurious patch attribution-energy aggregate chart";

  document.querySelector("#localization-title").textContent = model.label + " localization";
  document.querySelector("#faithfulness-title").textContent = model.label + " faithfulness";
  document.querySelector("#spurious-title").textContent = model.label + " patched model";

  if (!fullSummary) return;
  const train = fullSummary.train[model.variant];
  setMetric("val-accuracy", train.val_accuracy.toFixed(3));
  setMetric("val-f1", train.val_macro_f1.toFixed(3));
  setMetric("best-pointing", bestAttributionPointing(fullSummary, model.variant).toFixed(3));
  const ig = mean(
    fullSummary.randomization[model.variant].integrated_gradients.all.abs_spearman,
  );
  setMetric("ig-randomization", ig.toFixed(3));
  setMetric("ig-randomization-compact", ig.toFixed(3));
  setMetric("spurious-spread", "≤" + spuriousSpread(fullSummary, model).toFixed(3));
}

async function loadEvidence() {
  const status = document.querySelector("#data-status");
  try {
    const [summaryResponse, canaryResponse] = await Promise.all([
      fetch("data/summary.json"),
      fetch("data/cuda-resume-canary.json"),
    ]);
    if (!summaryResponse.ok || !canaryResponse.ok) {
      throw new Error("Evidence response was not successful.");
    }
    const [summary, canary] = await Promise.all([
      summaryResponse.json(),
      canaryResponse.json(),
    ]);
    if (summary.schema_version !== 1 || summary.experiment !== "full") {
      throw new Error("Unexpected summary contract.");
    }
    if (canary.schema_version !== 1 || canary.status !== "PASS") {
      throw new Error("Unexpected canary contract.");
    }

    fullSummary = summary;
    const center = mean(summary.localization.cnn.center.all.pointing_rate);
    setMetric("center-pointing", center.toFixed(3));

    const canaryMap = {
      head: "head_state_exact",
      optimizer: "optimizer_state_exact",
      scaler: "grad_scaler_state_exact",
      metrics: "stable_metrics_exact",
    };
    Object.entries(canaryMap).forEach(([elementName, evidenceName]) => {
      const element = document.querySelector("[data-canary='" + elementName + "']");
      element.textContent = canary.comparisons[evidenceName] ? "exact" : "difference";
    });

    const selected = document.querySelector("[data-model][aria-pressed='true']");
    selectModel(selected ? selected.dataset.model : "cnn");
    status.textContent = "Canonical full-scale JSON loaded and validated in this page.";
  } catch (error) {
    status.textContent =
      "Committed fallback shown; machine-readable evidence could not be loaded.";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("[data-model]").forEach((button) => {
    button.addEventListener("click", () => selectModel(button.dataset.model));
  });
  loadEvidence();
});

