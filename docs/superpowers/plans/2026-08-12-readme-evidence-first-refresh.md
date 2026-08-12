# Evidence-First README Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refine the bilingual repository landing page into a Traditional-Chinese-first, evidence-led portfolio README and replace its obsolete generated hero with a current 1440×900 showcase capture.

**Architecture:** Keep `README.md` as the primary Traditional Chinese entry point and `README_en.md` as a complete secondary edition. Treat the machine-generated result blocks as immutable, consolidate the surrounding narrative and Mermaid presentation, and separate the deterministic social-card renderer from the browser-captured README hero so future asset generation cannot overwrite the current UI screenshot.

**Tech Stack:** Markdown, Mermaid, Python 3.11, Pillow, Playwright browser capture, pytest, Ruff, mypy, Git.

## Global Constraints

- Preserve the exact content between `<!-- RESULTS:BEGIN -->` and `<!-- RESULTS:END -->` in both READMEs.
- Use Traditional Chinese (`zh-TW`) as the primary language; retain established technical terms in English without repetitive parenthetical translations.
- Preserve the three verified conclusions: Center Prior wins the Pointing Game comparison, Integrated Gradients fails Model Randomization, and Spurious Patch is a negative result.
- Never describe Localization as causal Faithfulness or the fixed 500-sample Attribution subset as the complete test split.
- The screenshot is presentation-only evidence derived from the allowlisted static showcase; it must not contain samples, weights, checkpoints, secrets, or browser chrome.
- Do not add a remote, push, create a PR, publish a release, deploy, or upload artifacts.
- Keep author and committer identity `kuotunyu <61350295+kuotunyu@users.noreply.github.com>` and add no contributor trailers.
- Make small English commits without amend, squash, reset, or history rewrite.

---

### Task 1: Lock the evidence-first README contract

**Files:**
- Modify: `tests/test_release_verifier.py`
- Modify: `README.md`
- Modify: `README_en.md`

**Interfaces:**
- Consumes: `assets/portfolio/hero.png`, `results/derived/summary.md`, and the existing release-verifier link checks.
- Produces: `README.md` headed `# Vision XAI Reliability Lab`, a compact navigation row, one Mermaid diagram, and `## Release 狀態`; `README_en.md` mirrors the structure under `## Release status`.

- [ ] **Step 1: Write the failing README-structure tests**

Add focused assertions to `tests/test_release_verifier.py`:

```python
def test_readmes_use_the_evidence_first_information_architecture() -> None:
    primary = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    secondary = (REPO_ROOT / "README_en.md").read_text(encoding="utf-8")

    assert primary.startswith("# Vision XAI Reliability Lab\n")
    assert "[成果展示]" in primary
    assert "[快速開始](#快速開始)" in primary
    assert "[實驗證據](ARTIFACTS.md)" in primary
    assert "[Model Card](MODEL_CARD.md)" in primary
    assert "[English](README_en.md)" in primary
    assert "## Release 狀態" in primary
    assert "## 專案進度" not in primary

    assert "[正體中文](README.md)" in secondary
    assert "## Release status" in secondary
    assert "## Project status" not in secondary

    for text in (primary, secondary):
        assert "FastAPI-0.110" not in text
        assert text.count("```mermaid") == 1
```

Update `test_readmes_lead_with_portfolio_evidence` to use `## Release 狀態` and `## Release status` as the status headings.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```powershell
uv run --no-sync pytest tests/test_release_verifier.py -k "readmes" -q
```

Expected: FAIL because the current title, FastAPI badge, two Mermaid diagrams, and old progress headings violate the new contract.

- [ ] **Step 3: Implement the compact bilingual README hierarchy**

Edit only the human-authored sections outside the result markers:

```markdown
# Vision XAI Reliability Lab

[CI badge] [Python badge] [PyTorch badge] [License badge]

這是一套以可靠性為核心的 Vision XAI benchmark，針對 ConvNeXt-Tiny 與 ViT-B/16，分開檢驗 Heatmap 的 Localization、Faithfulness、Model Randomization、Flip Consistency 與 Spurious Patch 行為。

重點不在產生更漂亮的 Heatmap，而在回答：它是否真的依賴模型學到的證據，以及結論能否被重算、稽核與反駁。

[成果展示](https://kuotunyu.github.io/vision-xai-reliability-lab/) · [快速開始](#快速開始) · [實驗證據](ARTIFACTS.md) · [Model Card](MODEL_CARD.md) · [English](README_en.md)
```

Apply the approved terminology throughout the primary narrative: retain `Vision XAI`, `Attribution`, `Heatmap`, `Localization`, `Faithfulness`, `Model Randomization`, `Flip Consistency`, `Spurious Patch`, `Center Prior`, `Pointing Game`, `Integrated Gradients`, `Head Randomization`, `CUDA AMP`, `SHA-256 manifest`, `full-scale L4 run`, and `test split` without duplicate translations. Replace the two diagrams with one compact evidence-flow Mermaid and replace the all-complete stage table with four release-status bullets covering verified full-scale evidence, the CUDA resume canary, CPU/release gates, and the local-only publication boundary. Mirror the information architecture in concise English.

- [ ] **Step 4: Confirm the generated evidence blocks are byte-identical to their source**

Run:

```powershell
uv run --no-sync pytest tests/test_release_verifier.py -k "readme_result_blocks or readmes" -q
```

Expected: PASS; the result-marker verifier proves both README result blocks still match `results/derived/summary.md` exactly.

- [ ] **Step 5: Commit the README hierarchy**

```powershell
git add README.md README_en.md tests/test_release_verifier.py
git commit -m "Refine evidence-first README"
```

---

### Task 2: Prevent the asset generator from overwriting the browser capture

**Files:**
- Modify: `tools/portfolio_assets.py`
- Modify: `tests/test_portfolio_assets.py`
- Modify: `tests/test_release_verifier.py`

**Interfaces:**
- Consumes: `extract_portfolio_metrics(root: Path) -> PortfolioMetrics` and the existing Pillow social-card renderer.
- Produces: `render_social_preview(root: Path, output_dir: Path) -> Path`; the committed `hero.png` is independently checked as a 1440×900 showcase screenshot.

- [ ] **Step 1: Write failing renderer and screenshot-contract tests**

Replace the old two-image renderer test with:

```python
from tools.portfolio_assets import render_social_preview


def test_render_social_preview_does_not_overwrite_readme_hero(tmp_path: Path) -> None:
    hero = tmp_path / "hero.png"
    hero.write_bytes(b"browser capture")

    social = render_social_preview(REPO_ROOT, tmp_path)

    assert social == tmp_path / "social-preview.png"
    assert hero.read_bytes() == b"browser capture"
    with Image.open(social) as image:
        assert image.size == (1280, 640)
```

Add this release-artifact assertion:

```python
def test_readme_hero_is_a_current_showcase_capture() -> None:
    with Image.open(REPO_ROOT / "assets" / "portfolio" / "hero.png") as image:
        assert image.size == (1440, 900)

    manifest = json.loads(
        (REPO_ROOT / "release" / "artifact-manifest.json").read_text(encoding="utf-8")
    )
    entry = next(item for item in manifest["artifacts"] if item["path"] == "assets/portfolio/hero.png")
    assert entry["role"] == "browser capture of the allowlisted static showcase"
```

- [ ] **Step 2: Run focused tests and confirm RED**

Run:

```powershell
uv run --no-sync pytest tests/test_portfolio_assets.py tests/test_release_verifier.py -k "social_preview or current_showcase_capture" -q
```

Expected: collection or assertion failure because `render_social_preview` is not defined and the committed hero is still 1600×900.

- [ ] **Step 3: Narrow the deterministic asset renderer to the social card**

In `tools/portfolio_assets.py`, remove `_render_hero`, replace `render_portfolio_assets` with:

```python
def render_social_preview(root: Path, output_dir: Path) -> Path:
    root = root.resolve()
    output_dir = output_dir.resolve()
    if _is_relative_to(output_dir, root / "results"):
        raise ValueError("portfolio assets must not be written beneath results/")

    output_dir.mkdir(parents=True, exist_ok=True)
    social = output_dir / "social-preview.png"
    _render_social(extract_portfolio_metrics(root)).save(social, format="PNG", optimize=True)
    return social
```

Update `main()` to write and report only `social-preview.png`. Preserve metric extraction, palette, typography, and CLI safety rules.

- [ ] **Step 4: Run the renderer tests**

```powershell
uv run --no-sync pytest tests/test_portfolio_assets.py -q
```

Expected: PASS, proving social regeneration leaves the README screenshot untouched.

- [ ] **Step 5: Commit the asset-boundary change**

```powershell
git add tools/portfolio_assets.py tests/test_portfolio_assets.py tests/test_release_verifier.py
git commit -m "Separate showcase capture from social card"
```

---

### Task 3: Capture and register the current showcase

**Files:**
- Modify: `assets/portfolio/hero.png`
- Modify: `ARTIFACTS.md`
- Modify: `release/artifact-manifest.json`

**Interfaces:**
- Consumes: the output of `tools/build_showcase.py` and the current `showcase/index.html`, `showcase/styles.css`, and `showcase/app.js`.
- Produces: a no-browser-chrome 1440×900 PNG whose SHA-256 and byte size are registered in the release manifest.

- [ ] **Step 1: Build the allowlisted static showcase in an ASCII-only temporary directory**

Run:

```powershell
$captureRoot = Join-Path $env:TEMP ("vx-readme-hero-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -LiteralPath $captureRoot | Out-Null
uv run --no-sync python tools/build_showcase.py --root . --output $captureRoot
```

Expected: the builder reports 18 allowlisted files and no weights, samples, checkpoints, or runtime data.

- [ ] **Step 2: Capture the loaded page at the approved viewport**

First run the required helper check:

```powershell
python C:\Users\3Hml\.codex\skills\webapp-testing\scripts\with_server.py --help
```

Then serve `$captureRoot`, open `http://127.0.0.1:8765/` with Playwright at viewport `1440x900`, wait until the three `.finding` cards are visible and both JSON requests complete, and save a viewport screenshot to `assets/portfolio/hero.png`. The capture script must call:

```python
await page.set_viewport_size({"width": 1440, "height": 900})
await page.goto("http://127.0.0.1:8765/", wait_until="networkidle")
await page.locator(".finding").nth(2).wait_for(state="visible")
await page.screenshot(path="assets/portfolio/hero.png", full_page=False)
```

- [ ] **Step 3: Inspect the screenshot visually**

Open `assets/portfolio/hero.png` with the image viewer and confirm it contains the title, run scope, all three finding cards, no browser chrome, no clipped labels, and no sensitive paths or data samples. If any condition fails, adjust only the viewport/loading wait and recapture.

- [ ] **Step 4: Record provenance and refresh the hash manifest**

Add this concise provenance note to `ARTIFACTS.md`:

```markdown
### README showcase capture

`assets/portfolio/hero.png` is a 1440×900 browser capture of the allowlisted static showcase after both public JSON artifacts load. It is presentation-only, contains no input samples or model weights, and is protected by `release/artifact-manifest.json`.
```

Calculate the PNG byte size and SHA-256 with `Get-Item` and `Get-FileHash`, then use `apply_patch` to update only the `assets/portfolio/hero.png` entry in `release/artifact-manifest.json`, including the role `browser capture of the allowlisted static showcase`.

- [ ] **Step 5: Run artifact and showcase verification**

```powershell
uv run --no-sync pytest tests/test_release_verifier.py tests/test_showcase.py tests/test_portfolio_assets.py -q
uv run --no-sync python tools/verify_release.py --root .
uv run --no-sync python tools/build_showcase.py --root . --output .artifacts/showcase-final
uv run --no-sync python tools/audit_showcase.py --site-dir .artifacts/showcase-final
```

Expected: all tests and release checks PASS; the static build still contains exactly the allowlisted public surface.

- [ ] **Step 6: Commit the current showcase capture**

```powershell
git add assets/portfolio/hero.png ARTIFACTS.md release/artifact-manifest.json
git commit -m "Refresh README showcase capture"
```

---

### Task 4: Run the release-candidate gates on the committed tree

**Files:**
- Verify only; no source changes unless a gate exposes a release blocker.

**Interfaces:**
- Consumes: committed `main` after Tasks 1–3.
- Produces: fresh local evidence for formatting, typing, tests, package contents, public showcase contents, privacy, authorship, and Git cleanliness.

- [ ] **Step 1: Verify formatting, lint, typing, and the full CPU suite**

```powershell
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
uv run --no-sync mypy --strict src app tools tests
uv run --no-sync pytest -q
```

Expected: every command exits 0 without GPU, datasets, weights, or secrets.

- [ ] **Step 2: Build and audit distribution artifacts**

```powershell
uv build
uv run --no-sync python tools/audit_distribution.py --dist-dir dist
```

Expected: sdist and wheel contain only the intended package surface and no private/runtime files.

- [ ] **Step 3: Run the isolated wheel smoke**

Create a temporary virtual environment outside the repository, install the newly built wheel without repository import leakage, and run:

```powershell
python -c "import vision_xai; print(vision_xai.__version__)"
vision-xai --help
```

Expected: both commands exit 0 from outside the repository.

- [ ] **Step 4: Re-run release, secret, path, authorship, and Git-state audits**

```powershell
uv run --no-sync python tools/verify_release.py --root .
git log --format="%h%x09%an <%ae>%x09%cn <%ce>%x09%B" --all
git remote -v
git status --short --branch
git rev-list --left-right --count origin/main...main
```

Expected: release verifier PASS; every author/committer is the approved identity; no AI/co-author trailers, private paths, secrets, or oversized tracked files; only the pre-existing `origin` is configured; the worktree is clean and local `main` is ahead only by the deliberate unpublished commits.

- [ ] **Step 5: Report the local-only handoff**

Summarize the README structure, refreshed screenshot, exact commits, test results, artifact/privacy evidence, remaining limitations, recommended GitHub topics, and the owner-only push/action list. Do not push or modify external state.
