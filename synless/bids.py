"""BIDS (Dai et al., EMNLP Findings 2025), with optional context capacities.

Source: https://aclanthology.org/2025.findings-emnlp.373/ §4, Algorithm 1.
Rows are candidates; columns are validation instances, NOT generator families.
"""

import torch


def normalize_columns(matrix):
    matrix = torch.as_tensor(matrix, dtype=torch.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or not matrix.shape[1] or not torch.isfinite(matrix).all():
        raise ValueError("Need a finite candidate-by-validation attribution matrix")
    mean = matrix.mean(0)
    std = matrix.std(0, correction=1)
    active = std > 1e-12
    if not active.any():
        raise ValueError("All validation columns have zero influence variance")
    normalized = (matrix[:, active] - mean[active]) / std[active]
    return normalized, {"column_mean": mean.tolist(), "column_std": std.tolist(),
                        "active_columns": active.nonzero().flatten().tolist()}


def greedy_select(matrix, budget, *, normalize=True, iterative=True, groups=None, capacity=None):
    """Max_j(A_ij - selected_mean_j), removing each chosen row.

    Algorithm 1 leaves the empty-set average unspecified. We explicitly use a
    zero vector at the first iteration; exact ties choose the first input row.
    Optional per-context capacity is a SynLess adaptation for matched exposure.
    """
    matrix = torch.as_tensor(matrix, dtype=torch.float64)
    if matrix.ndim != 2 or not torch.isfinite(matrix).all() or not 0 < budget <= len(matrix):
        raise ValueError("Invalid matrix or selection budget")
    if normalize:
        matrix, _ = normalize_columns(matrix)
    if (groups is None) != (capacity is None):
        raise ValueError("Supply both groups and capacity, or neither")
    if groups is not None and (len(groups) != len(matrix) or capacity < 1):
        raise ValueError("Invalid context capacity")
    available = torch.ones(len(matrix), dtype=torch.bool)
    selected, counts = [], {}
    selected_sum = torch.zeros(matrix.shape[1], dtype=matrix.dtype)
    for _ in range(budget):
        if not available.any():
            raise ValueError("Budget exceeds eligible context capacities")
        reference = selected_sum / len(selected) if selected and iterative else torch.zeros_like(selected_sum)
        utility = (matrix - reference).max(1).values.masked_fill(~available, -torch.inf)
        index = int(utility.argmax())
        selected.append(index)
        selected_sum += matrix[index]
        available[index] = False
        if groups is not None:
            group = groups[index]
            counts[group] = counts.get(group, 0) + 1
            if counts[group] >= capacity:
                for i, g in enumerate(groups):
                    if g == group:
                        available[i] = False
    return selected


def matrix_selections(rows, column_tasks, per_context):
    """Compare selectors on exactly the same head-space attribution matrix."""
    matrix = torch.tensor([r["head_influence"] for r in rows], dtype=torch.float64)
    if matrix.shape[1] != len(column_tasks):
        raise ValueError("Attribution columns and task labels do not match")
    groups = [r["batch"] for r in rows]
    budget = len(set(groups)) * per_context
    task_ids = sorted(set(column_tasks))
    task_mean = torch.stack([matrix[:, [i for i, t in enumerate(column_tasks) if t == task]].mean(1)
                             for task in task_ids], dim=1)
    selectors = {
        "less_head_task_max": greedy_select(task_mean, budget, normalize=False, iterative=False, groups=groups, capacity=per_context),
        "less_head_mean": greedy_select(matrix.mean(1, keepdim=True), budget, normalize=False, iterative=False, groups=groups, capacity=per_context),
        "bids": greedy_select(matrix, budget, groups=groups, capacity=per_context),
        "bids_no_normalization": greedy_select(matrix, budget, normalize=False, groups=groups, capacity=per_context),
        "normalized_instance_max": greedy_select(matrix, budget, iterative=False, groups=groups, capacity=per_context),
    }
    # Also retain unconstrained paper selectors as a ranking diagnostic. Training
    # uses the explicitly named context-capacity adaptation for matched batches.
    unconstrained = {
        "bids": greedy_select(matrix, budget),
        "less_task_max": greedy_select(task_mean, budget, normalize=False, iterative=False),
    }
    normalized, norm_info = normalize_columns(matrix)
    diagnostics = {"normalization": norm_info, "tasks": task_ids,
                   "unconstrained_selections": {k: [rows[i]["id"] for i in v] for k, v in unconstrained.items()},
                   "pool_average_influence": matrix.mean(0).tolist(), "selectors": {}}
    for name, indices in selectors.items():
        highest = task_mean[indices].argmax(1)
        diagnostics["selectors"][name] = {
            "task_with_highest_influence_counts": {t: int((highest == i).sum()) for i, t in enumerate(task_ids)},
            "selected_mean_normalized_influence": normalized[indices].mean(0).tolist(),
            "mean_influence_by_task": {t: float(task_mean[indices, i].mean()) for i, t in enumerate(task_ids)},
        }
    return selectors, diagnostics
