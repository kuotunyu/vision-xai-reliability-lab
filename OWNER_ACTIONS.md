# Owner actions after local release-candidate approval

The public GitHub repository, `origin` remote, GitHub Actions, and Pages source
already exist. This local release workflow does not push, deploy, tag, or change
any other remote state.

## GitHub update checklist

1. Review the unpublished local commits:

   ```text
   git log origin/main..main --oneline
   ```

2. When satisfied, push `main` yourself:

   ```text
   git push origin main
   ```

3. Confirm that CI and Pages workflows pass for the pushed head.
   Then verify the README hero, local links, license, Pages model switch, and
   public JSON/document links in the GitHub UI.
4. Keep the repository **About** panel in Traditional Chinese (`zh-TW`):
   - Repository description: `以可靠性為核心的 XAI benchmark：比較 ConvNeXt-Tiny 與 ViT-B/16，檢驗 Localization、Faithfulness、Model Randomization 與可重現 CUDA resume 證據。`
   - Topics: `computer-vision`, `explainable-ai`, `trustworthy-ai`,
     `model-interpretability`, `model-evaluation`, `reproducibility`, and `pytorch`.
   - Website: `https://kuotunyu.github.io/vision-xai-reliability-lab/`.
5. In **Settings → General → Social preview**, upload
   `assets/portfolio/social-preview.png` if the current preview is missing or
   outdated.
6. Pin the repository on the GitHub profile after the pushed checks pass.

The GitHub README, About description, Pages showcase, and Gradio workbench use
Traditional Chinese (`zh-TW`) as the primary language while retaining English
technical terms. Pages is a static results explorer; Gradio adds a separate
local-model layer that remains unavailable until the owner supplies a compatible
checkpoint. The CPU Docker image includes only the canonical aggregate summary,
CUDA canary, and six aggregate figures—never the dataset, weights, or checkpoints.

## Release decision

Do not create a tag or GitHub Release until the pushed commit passes CI and the
Pages evidence links have been checked. The local candidate intentionally has
neither. The Pages site is a static results showcase, not a live inference demo;
it needs no API deployment, GPU, dataset, model weight, account token, or project
secret.

Do not rerun or replace the committed full-scale L4 aggregate merely to publish
the repository. Any future broader claims require separately recorded evidence.
