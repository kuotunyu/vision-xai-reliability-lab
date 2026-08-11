"""Command-line interface. Later stages extend this surface; unimplemented
commands are registered as stubs that exit with code 2."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from vision_xai.errors import VisionXAIError
from vision_xai.logging_utils import setup_logging

logger = logging.getLogger(__name__)

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Vision XAI Reliability Lab")
data_app = typer.Typer(no_args_is_help=True, help="Dataset preparation commands.")
app.add_typer(data_app, name="data")

_STUB_EXIT_CODE = 2


@data_app.command("prepare")
def data_prepare(
    config: Annotated[Path, typer.Option("--config", help="Path to a YAML config file.")],
    resume: Annotated[
        bool, typer.Option("--resume", help="Continue an interrupted preparation run.")
    ] = False,
    max_items: Annotated[
        int | None,
        typer.Option("--max-items", min=1, help="Hash at most N new samples, then checkpoint."),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", help="Enable debug logging.")] = False,
) -> None:
    """Download/verify Oxford-IIIT Pet, build the split manifest, fingerprint,
    and spurious-patch assignments."""
    setup_logging("DEBUG" if verbose else None)
    from vision_xai.config import load_config
    from vision_xai.data.prepare import prepare_data

    try:
        cfg = load_config(config)
        result = prepare_data(cfg, resume=resume, max_items=max_items)
    except KeyboardInterrupt:
        typer.echo(
            "interrupted — progress up to the last checkpoint is saved; rerun with --resume",
            err=True,
        )
        raise typer.Exit(130) from None
    except VisionXAIError as exc:
        logger.error("%s", exc)
        raise typer.Exit(1) from exc
    if not result.completed:
        typer.echo(
            f"checkpointed after {result.num_processed_this_run} newly hashed samples "
            f"({result.num_total} total); continue with --resume"
        )
        return
    typer.echo(
        f"data prepare complete: {result.counts_by_split} "
        f"(content_sha256={result.content_sha256}, {result.elapsed_seconds:.1f}s)"
    )


@app.command("self-check")
def self_check(
    quiet: Annotated[
        bool, typer.Option("--quiet", help="Only report failures (for healthchecks).")
    ] = False,
) -> None:
    """CPU smoke test: torch matmul, PIL round-trip, config parsing, runtime versions."""
    setup_logging("ERROR" if quiet else None)
    import io as std_io

    import torch
    from PIL import Image

    import vision_xai
    from vision_xai.config import load_config

    try:
        matrix = torch.rand(64, 64)
        product = matrix @ matrix
        if product.shape != (64, 64):
            raise VisionXAIError(f"unexpected matmul result shape {tuple(product.shape)}")

        buffer = std_io.BytesIO()
        Image.new("RGB", (32, 32), color=(255, 0, 255)).save(buffer, format="PNG")
        buffer.seek(0)
        with Image.open(buffer) as reopened:
            if reopened.size != (32, 32):
                raise VisionXAIError(f"unexpected PIL round-trip size {reopened.size}")

        smoke_config = Path("configs") / "smoke.yaml"
        if smoke_config.is_file():
            load_config(smoke_config)
        else:
            logger.info("configs/smoke.yaml not found in cwd; skipping config check")

        vision_xai.check_runtime()
    except VisionXAIError as exc:
        typer.echo(f"self-check FAILED: {exc}", err=True)
        raise typer.Exit(1) from exc
    if not quiet:
        typer.echo(f"self-check OK (vision-xai {vision_xai.__version__})")


def _stub(command: str, stage: str) -> None:
    typer.echo(f"'{command}' is not implemented yet — it arrives in {stage}.", err=True)
    raise typer.Exit(_STUB_EXIT_CODE)


@app.command()
def train(
    model: Annotated[str, typer.Option("--model", help="cnn or vit")],
    config: Annotated[Path, typer.Option("--config", help="Path to a YAML config file.")],
    patched: Annotated[
        bool,
        typer.Option("--patched", help="Train with the spurious corner patch (train_correlated)."),
    ] = False,
    resume: Annotated[
        bool, typer.Option("--resume", help="Continue from the last checkpoint.")
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", help="Enable debug logging.")] = False,
) -> None:
    """Train a classifier head (backbone frozen) and checkpoint every epoch."""
    setup_logging("DEBUG" if verbose else None)
    from vision_xai.config import load_config
    from vision_xai.models.factory import MODEL_NAMES
    from vision_xai.train.loop import train_model

    if model not in MODEL_NAMES:
        typer.echo(f"--model must be one of {', '.join(MODEL_NAMES)}", err=True)
        raise typer.Exit(1)
    try:
        cfg = load_config(config)
        result = train_model(cfg, model, patched=patched, resume=resume)
    except VisionXAIError as exc:
        logger.error("%s", exc)
        raise typer.Exit(1) from exc
    peak = f"{result.peak_vram_mb}MB" if result.peak_vram_mb is not None else "n/a (cpu)"
    typer.echo(
        f"train complete: {result.variant} epochs={result.total_epochs_completed} "
        f"val_acc={result.final_val_accuracy:.4f} val_f1={result.final_val_macro_f1:.4f} "
        f"val_ece={result.final_val_ece:.4f} "
        f"device={result.device} peak_vram={peak} elapsed={result.elapsed_seconds:.1f}s"
    )


@app.command()
def explain(
    model: Annotated[str, typer.Option("--model", help="cnn or vit")],
    method: Annotated[
        str,
        typer.Option(
            "--method",
            help="gradcam | integrated_gradients | occlusion | random | uniform | center",
        ),
    ],
    config: Annotated[Path, typer.Option("--config", help="Path to a YAML config file.")],
    patched: Annotated[
        bool, typer.Option("--patched", help="Use the patched-trained model variant.")
    ] = False,
    test_variant: Annotated[
        str,
        typer.Option(
            "--test-variant",
            help="clean | correlated | no_patch | counter_correlated (test images).",
        ),
    ] = "clean",
    resume: Annotated[
        bool, typer.Option("--resume", help="Skip samples that already have attributions.")
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", help="Enable debug logging.")] = False,
) -> None:
    """Compute attribution maps for a trained model over the configured split."""
    setup_logging("DEBUG" if verbose else None)
    from vision_xai.config import load_config
    from vision_xai.explain.runner import TEST_VARIANTS, run_explanations
    from vision_xai.models.factory import MODEL_NAMES

    if model not in MODEL_NAMES:
        typer.echo(f"--model must be one of {', '.join(MODEL_NAMES)}", err=True)
        raise typer.Exit(1)
    if test_variant not in TEST_VARIANTS:
        typer.echo(f"--test-variant must be one of {', '.join(TEST_VARIANTS)}", err=True)
        raise typer.Exit(1)
    try:
        cfg = load_config(config)
        result = run_explanations(
            cfg, model, method, patched=patched, test_variant=test_variant, resume=resume
        )
    except VisionXAIError as exc:
        logger.error("%s", exc)
        raise typer.Exit(1) from exc
    typer.echo(
        f"explain complete: {result.variant}/{result.method} ({result.test_variant}) "
        f"{result.num_processed_this_run} newly computed of {result.num_total} total "
        f"in {result.elapsed_seconds:.1f}s -> {result.out_dir}"
    )


@app.command()
def evaluate(
    config: Annotated[Path, typer.Option("--config", help="Path to a YAML config file.")],
    verbose: Annotated[bool, typer.Option("--verbose", help="Enable debug logging.")] = False,
) -> None:
    """Evaluate every completed explanation run (incremental — safe to rerun)."""
    setup_logging("DEBUG" if verbose else None)
    from vision_xai.config import load_config
    from vision_xai.eval.runner import evaluate_all

    try:
        cfg = load_config(config)
        result = evaluate_all(cfg)
    except VisionXAIError as exc:
        logger.error("%s", exc)
        raise typer.Exit(1) from exc
    typer.echo(
        f"evaluate complete: {result.rows_written} rows this run "
        f"in {result.elapsed_seconds:.1f}s -> {result.out_dir}"
    )


@app.command()
def report(
    config: Annotated[Path, typer.Option("--config", help="Path to a YAML config file.")],
    verbose: Annotated[bool, typer.Option("--verbose", help="Enable debug logging.")] = False,
) -> None:
    """Aggregate raw results into summary.json + figures and update the READMEs."""
    setup_logging("DEBUG" if verbose else None)
    from vision_xai.config import load_config
    from vision_xai.report.build import build_report

    try:
        cfg = load_config(config)
        result = build_report(cfg)
    except VisionXAIError as exc:
        logger.error("%s", exc)
        raise typer.Exit(1) from exc
    typer.echo(
        f"report complete: {result.summary_path} | {len(result.figure_paths)} figures | "
        f"READMEs updated: {', '.join(str(p) for p in result.readmes_updated) or 'none'} "
        f"({result.elapsed_seconds:.1f}s)"
    )


@app.command()
def serve(
    config: Annotated[Path, typer.Option("--config", help="Path to a YAML config file.")] = Path(
        "configs"
    )
    / "smoke.yaml",
    host: Annotated[
        str | None, typer.Option("--host", help="Bind address (overrides serve.host).")
    ] = None,
    port: Annotated[int | None, typer.Option("--port", help="Port (overrides serve.port).")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", help="Enable debug logging.")] = False,
) -> None:
    """Serve the FastAPI app (with the Gradio demo at /demo when gradio is installed).

    Bare `serve` (no --config) starts the smoke config, matching the precomputed
    CPU demo requirement — it needs no dataset or GPU.
    """
    setup_logging("DEBUG" if verbose else None)
    from vision_xai.config import load_config

    try:
        cfg = load_config(config)
        try:
            from app.api import create_app
        except ImportError as exc:
            raise VisionXAIError(
                "the serving layer requires the repository's app/ package on the "
                f"Python path (run from the repository root): {exc}"
            ) from exc
        import uvicorn

        application = create_app(cfg)
    except VisionXAIError as exc:
        logger.error("%s", exc)
        raise typer.Exit(1) from exc
    uvicorn.run(
        application,
        host=host if host is not None else cfg.serve.host,
        port=port if port is not None else cfg.serve.port,
        log_level="debug" if verbose else "info",
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
