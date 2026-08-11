# Owner actions after local release-candidate approval

No remote, deployment, model-hosting, or release state is changed by the local
release workflow. The repository owner can perform these actions later:

1. Review the three headline findings and the evidence boundary in
   `README.md`, `MODEL_CARD.md`, `DATA_CARD.md`, and `ARTIFACTS.md`.
2. Run the documented CPU release gates from a clean checkout and confirm that
   the hosted CI passes without datasets, weights, a GPU, or secrets.
3. Create the public remote and push `main` manually. Do not add the raw Oxford
   dataset, checkpoints, pretrained weights, or ignored `.artifacts/` outputs.
4. Verify rendered README links, figures, license metadata, API documentation,
   and the `/demo` route on the chosen hosting platform.
5. Only after the pushed commit passes CI, decide whether to create a version
   tag or release. The local candidate intentionally has neither.

Do not rerun or replace the committed full-scale L4 aggregate merely to publish
the repository. Any future broader claims require separately recorded evidence.
