# Owner actions after local release-candidate approval

No remote, deployment, model-hosting, tag, or release state is changed by the
local release workflow. The repository owner can perform the steps below after
reviewing the final local commit.

## GitHub repository setup

1. Create an empty **public** repository named
   `vision-xai-reliability-lab`. Do not initialize it with another README,
   license, or `.gitignore`.
2. Configure the repository **About** panel:
   - Repository description: `以可靠性為核心的 XAI benchmark：比較 ConvNeXt 與
     ViT 的 localization、faithfulness、sanity checks 與可重現 CUDA resume
     證據。`
   - Topics: `computer-vision`, `explainable-ai`, `trustworthy-ai`, `pytorch`,
     `machine-learning`, `model-evaluation`, `reproducibility`, `fastapi`, and
     `gradio`.
   - Website: `https://kuotunyu.github.io/vision-xai-reliability-lab/` after
     Pages succeeds.
3. In **Settings → General → Social preview**, upload
   `assets/portfolio/social-preview.png`.
4. In **Settings → Pages**, select **GitHub Actions** as the build and
   deployment source.
5. Add the GitHub remote locally and push `main` manually. Do not add the raw
   Oxford-IIIT Pet dataset, checkpoints, pretrained weights, or ignored
   `.artifacts/` outputs.
6. Confirm that both `ci` jobs pass and that the `pages` workflow publishes the
   18-file allowlisted showcase. Verify the README hero, local links, license,
   Pages model switch, and public JSON/document links in the GitHub UI.
7. Pin the repository on the GitHub profile after those checks pass.

The GitHub README, About description, Pages showcase, and Gradio workbench use
Traditional Chinese (`zh-TW`) as the primary language while retaining English
technical terms. Pages is a static results explorer; Gradio adds a separate
local-model layer that remains unavailable until the owner supplies a compatible
checkpoint. The CPU Docker image includes only the canonical aggregate summary,
CUDA canary, and six aggregate figures—never the dataset, weights, or checkpoints.

## Release decision

Only after the pushed commit passes CI should the owner decide whether to
create a version tag or GitHub Release. The local candidate intentionally has
neither. The Pages site is a static results showcase, not a live inference
demo; it needs no API deployment, GPU, dataset, model weight, account token, or
project secret.

Do not rerun or replace the committed full-scale L4 aggregate merely to publish
the repository. Any future broader claims require separately recorded evidence.
