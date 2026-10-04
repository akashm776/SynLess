"""Check local documentation links, notebook syntax, and archived pilot arithmetic.

Uses only the standard library. Does not download data or execute training cells.
"""

import ast
import json
import math
import re
import statistics
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]


def check_links():
    pages = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")),
             *sorted((ROOT / "results").rglob("*.md"))]
    for page in pages:
        content = page.read_text()
        assert content.count("```") % 2 == 0, f"Unclosed code fence: {page}"
        for target in re.findall(r"\]\(([^\s)]+)\)", content):
            parsed = urlsplit(target.strip("<>"))
            if parsed.scheme or not parsed.path:
                continue
            resolved = (page.parent / unquote(parsed.path)).resolve()
            assert resolved.is_relative_to(ROOT), f"Link escapes repo: {page}: {target}"
            assert resolved.exists(), f"Broken local link: {page}: {target}"
    return len(pages)


def check_notebooks():
    notebooks = sorted((ROOT / "colabs").glob("*.ipynb"))
    for path in notebooks:
        notebook = json.loads(path.read_text())
        assert notebook["nbformat"] == 4
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]), filename=f"{path}:{index}")
                assert not cell.get("outputs"), f"Committed notebook output: {path}:{index}"
    return len(notebooks)


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12), (actual, expected)


def check_results():
    archive = ROOT / "results" / "clip_cub_pilot_v1"
    for path in archive.rglob("*.json"):
        json.loads(path.read_text())
    manifest = json.loads((archive / "run.json").read_text())
    summary = json.loads((archive / "summary.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["identity"]["config"]["backend"] == "cub"
    rows = summary["paired_results"]
    assert len(rows) == 90
    keyed = {(r["seed"], r["starting_update"], r["arm"]): r for r in rows}
    assert len(keyed) == len(rows)
    for row in rows:
        native = keyed[(row["seed"], row["starting_update"], "native")]
        assert row["initial"] == native["initial"]
        assert row["updates"] == 20
        assert row["final"]["pool_size"] == 512
        assert row["auxiliary_exposures"] == (0 if row["arm"] == "native" else 160)
        for metric, delta in row["paired_delta"].items():
            close(delta, row["final"][metric] - native["final"][metric])
    assert len(summary["aggregate"]) == 30
    for aggregate in summary["aggregate"]:
        state, arm = aggregate["state_arm"].split(":")
        members = [r for r in rows if r["starting_update"] == int(state) and r["arm"] == arm]
        assert len(members) == aggregate["training_seeds"] == 3
        values = [r["paired_delta"]["native_loss"] for r in members]
        close(aggregate["mean_loss_delta"], statistics.mean(values))
        close(aggregate["seed_std_loss_delta"], statistics.stdev(values))
        for row in members:
            close(aggregate["per_seed_loss_delta"][str(row["seed"])], row["paired_delta"]["native_loss"])
        for prefix, metric_prefix in [("t2i", "text_to_image"), ("i2t", "image_to_text")]:
            for suffix in ["loss", "r1"]:
                close(aggregate[f"mean_{prefix}_{suffix}_delta"], statistics.mean(
                    r["paired_delta"][f"{metric_prefix}_{suffix}"] for r in members))
    # The README's paired OT-versus-uniform statement must remain reproducible.
    for seed in [789, 2026, 31415]:
        for state in [100, 500]:
            assert keyed[(seed, state, "ot")]["paired_delta"]["native_loss"] < keyed[
                (seed, state, "uniform")]["paired_delta"]["native_loss"]
    return len(rows)


if __name__ == "__main__":
    print(f"Checked {check_links()} Markdown files, {check_notebooks()} notebook(s), "
          f"and {check_results()} archived matched branches.")
