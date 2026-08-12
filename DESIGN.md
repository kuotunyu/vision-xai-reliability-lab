---
name: Vision XAI Reliability Lab
description: Evidence-first visual system for a trustworthy Computer Vision portfolio.
colors:
  canvas-ink: "#07131d"
  evidence-panel: "#0c212c"
  evidence-panel-strong: "#102a37"
  divider-steel: "#24414d"
  evidence-ivory: "#f3efe7"
  evidence-muted: "#adc0c8"
  measured-cyan: "#42d2e1"
  failure-coral: "#ff785f"
  verified-lime: "#a7e33f"
  focus-amber: "#f7c95c"
typography:
  display:
    fontFamily: "Noto Sans TC, Microsoft JhengHei, PingFang TC, sans-serif"
    fontSize: "clamp(2.7rem, 6vw, 6rem)"
    fontWeight: 800
    lineHeight: 0.98
    letterSpacing: "-0.035em"
  body:
    fontFamily: "Noto Sans TC, Microsoft JhengHei, PingFang TC, sans-serif"
    fontSize: "17px"
    fontWeight: 400
    lineHeight: 1.6
    letterSpacing: "normal"
  data-label:
    fontFamily: "Cascadia Code, Consolas, monospace"
    fontSize: "0.78rem"
    fontWeight: 650
    lineHeight: 1.35
    letterSpacing: "0.025em"
rounded:
  control: "10px"
  compact-panel: "12px"
  evidence-panel: "16px"
  hero-panel: "18px"
spacing:
  xs: "8px"
  sm: "12px"
  md: "16px"
  lg: "24px"
components:
  button-primary:
    backgroundColor: "{colors.measured-cyan}"
    textColor: "{colors.canvas-ink}"
    rounded: "{rounded.control}"
    padding: "12px 20px"
    height: "50px"
  evidence-chip:
    backgroundColor: "{colors.canvas-ink}"
    textColor: "{colors.measured-cyan}"
    typography: "{typography.data-label}"
    rounded: "999px"
    padding: "7px 11px"
  evidence-panel:
    backgroundColor: "{colors.evidence-panel}"
    textColor: "{colors.evidence-ivory}"
    rounded: "{rounded.evidence-panel}"
    padding: "20px"
---

# Design System: Vision XAI Reliability Lab

## Overview

**Creative North Star: "The Evidence Workbench"**

The interface should feel like a clear, carefully labeled instrument used to inspect an
experiment. Dense evidence, visible boundaries, and calm state colors carry the identity;
decoration never competes with the claims. Traditional Chinese (`zh-TW`) is the primary
reading language, while technical terms remain in their established English form.

The static showcase uses square editorial frames; the Gradio workbench uses gently rounded
interactive panels. Both surfaces share one dark palette, one typographic voice, and one
evidence-first reading order.

**Key Characteristics:**

- Dark navy work surface with warm off-white text.
- Cyan for measured evidence, coral for failed expectations, and lime for verified states.
- Large Traditional Chinese headings and compact monospace data labels.
- Dense, responsive grids with explicit claim boundaries.

## Colors

The palette is a low-glare technical field with three sparse semantic accents.

### Primary

- **Measured Cyan:** primary actions, selected controls, measurements, and active navigation.

### Secondary

- **Failure Coral:** failed sanity expectations, negative-result emphasis, and recoverable errors.
- **Verified Lime:** scoped pass states and successful reproducibility evidence.

### Neutral

- **Canvas Ink:** uninterrupted page background.
- **Evidence Panel:** primary content surface.
- **Evidence Panel Strong:** internal metric cells and selected controls.
- **Divider Steel:** one-pixel structural separation.
- **Evidence Ivory:** primary text and chart paper.
- **Evidence Muted:** explanations, metadata, and secondary labels.
- **Focus Amber:** keyboard focus only.

### Named Rules

**The Semantic Accent Rule.** Cyan measures, coral challenges, and lime verifies; do not use
these accents as arbitrary decoration.

**The Evidence Boundary Rule.** Every panel edge separates a claim, its evidence, or its scope.

## Typography

**Display Font:** Noto Sans TC (with Microsoft JhengHei, PingFang TC, and sans-serif fallbacks)

**Body Font:** Noto Sans TC (with the same Traditional Chinese fallbacks)

**Label/Mono Font:** Cascadia Code (with Consolas and monospace fallbacks)

**Character:** A single sturdy sans voice keeps mixed Chinese and English copy cohesive. Mono
is restricted to data, run metadata, version identifiers, and machine-readable references.

### Hierarchy

- **Display** (800, fluid 2.7–6rem, 0.98): product title and major showcase statements.
- **Headline** (800, fluid 2.5–4.6rem, 0.98): section conclusions.
- **Title** (700–800, 1.16–1.45rem, 1.35): claim and component headings.
- **Body** (400, 17px, 1.6): explanations, limitations, and recovery instructions; keep prose near 68ch.
- **Label** (650, 0.78rem, 0.025em): measured values and technical metadata only.

### Named Rules

**The Plain Technical Term Rule.** Keep `localization`, `faithfulness`, `checkpoint`, and related
terms in English inside natural Traditional Chinese sentences.

## Layout

Gradio content uses a wide 1480px ceiling with 24px page gutters and compact 12–24px internal
rhythm. Its first viewport flows from identity and scope to two tabs and a continuous three-part
finding band. The static showcase uses a narrower 72rem editorial measure and more separation
between narrative sections.

At 1080px, five-column evidence strips become two columns and figures wrap. At 760px, all core
evidence becomes a single column, gutters tighten to 12px, and the reading order remains
unchanged. Horizontal scrolling is not part of the system.

## Elevation & Depth

The system is flat by default. One-pixel borders, adjacent tonal surfaces, and chart paper create
depth; panels do not combine a border with a resting shadow. A soft shadow may appear on an
active hover control, but never as persistent card decoration.

### Named Rules

**The Flat Instrument Rule.** Resting evidence is separated by tone and rule, not elevation.

## Shapes

Interactive Gradio surfaces use restrained 10–18px radii: controls are tighter, content panels
are broader, and pills are reserved for short status chips. Static editorial frames remain square.
All structural borders are one pixel; semantic emphasis may use a bottom edge on a finding card.

## Components

### Buttons

- **Shape:** compact rounded control (10px) with a minimum 50px touch height.
- **Primary:** measured cyan on canvas ink with assertive 800-weight text.
- **Hover / Focus:** lighter cyan on hover; a three-pixel focus-amber outline with offset on keyboard focus.
- **Secondary:** transparent or panel-toned with a measured-cyan border.

### Chips

- **Style:** compact pill with a divider-steel border, canvas fill, cyan mono label, and 7px × 11px padding.
- **State:** status only; chips do not replace actions.

### Cards / Containers

- **Corner Style:** square on the static showcase, 16px on Gradio evidence panels.
- **Background:** evidence panel, with evidence panel strong for metric cells.
- **Shadow Strategy:** flat at rest.
- **Border:** one-pixel divider steel.
- **Internal Padding:** 20–24px for panels; 16px for dense metric cells.

### Inputs / Fields

- **Style:** Gradio controls inherit the panel palette, visible label, and 10px control radius.
- **Focus:** three-pixel focus-amber outline with a three-pixel offset.
- **Error / Disabled:** coral text names the problem and recovery; unavailable checkpoint controls are omitted in favor of a compact readiness state.

### Navigation

The product exposes two persistent top-level tabs: `實驗證據` and `本機模型`. Active state uses
ivory text, a panel fill, and a cyan bottom rule; focus remains visible independently of color.

### Finding Band

Three evidence conclusions share one continuous bordered band. Internal one-pixel dividers replace
three detached cards; on mobile the band becomes a vertical sequence without changing order.

## Do's and Don'ts

### Do:

- **Do** lead with versioned measurements, then state interpretation and scope.
- **Do** keep Traditional Chinese large enough to scan at desktop and mobile widths.
- **Do** use the semantic accents consistently and provide a text label for every state.
- **Do** fail closed with an em dash or a named error when evidence has not validated.

### Don't:

- **Don't** use gradients, glass effects, decorative grid overlays, or persistent card shadows.
- **Don't** use mono as a general-purpose technical costume or number sections that are not sequential.
- **Don't** add empty panels, unavailable controls, or hierarchy that does not help inspect evidence.
- **Don't** imply that localization proves causal faithfulness or that a tiny canary is full training.
