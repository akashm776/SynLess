import pytest
import torch

from synless.bids import normalize_columns, greedy_select, matrix_selections
from synless.attribution import directional_losses
from synless.core import native_loss


def test_iteration_prefers_missing_influence_direction():
    matrix = torch.tensor([[10., 0.], [9., 0.], [0., 8.], [0., 7.]])
    assert greedy_select(matrix, 2, normalize=False) == [0, 2]
    assert greedy_select(matrix, 2, normalize=False, iterative=False) == [0, 1]


def test_normalization_invariant_to_positive_column_scale_and_shift():
    matrix = torch.tensor([[2., 9.], [1., 8.], [4., 1.], [3., 2.]], dtype=torch.float64)
    original = greedy_select(matrix, 3)
    transformed = matrix * torch.tensor([50., .01]) + torch.tensor([-7., 100.])
    assert greedy_select(transformed, 3) == original
    norm, _ = normalize_columns(matrix)
    torch.testing.assert_close(norm.mean(0), torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(norm.std(0), torch.ones(2, dtype=torch.float64))


def test_degenerate_columns_are_excluded():
    matrix = torch.tensor([[2., 0.], [1., 0.], [4., 0.], [3., 0.]])
    normalized, info = normalize_columns(matrix)
    assert normalized.shape == (4, 1)
    assert info["active_columns"] == [0]
    with pytest.raises(ValueError, match="zero influence variance"):
        greedy_select(torch.ones(4, 2), 2)


def test_context_capacity_and_no_repeated_candidates():
    matrix = torch.tensor([[10., 3.], [9., 5.], [8., 1.], [0., 7.]])
    selected = greedy_select(matrix, 2, normalize=False, groups=[0, 0, 0, 1], capacity=1)
    assert selected == [0, 3]
    with pytest.raises(ValueError, match="capacities"):
        greedy_select(matrix, 3, groups=[0, 0, 0, 1], capacity=1)


def test_matrix_selectors_share_pool_and_respect_budget():
    rows = [{"id": str(i), "batch": i // 3, "head_influence": x} for i, x in enumerate(
        [[.1, .3], [.2, .2], [.4, -.1], [-.1, .5], [.2, .1], [.1, .4]])]
    selectors, diagnostics = matrix_selections(rows, ["t2i", "i2t"], 1)
    assert len(selectors) == 5
    for chosen in selectors.values():
        assert len(set(chosen)) == 2
        assert {rows[i]["batch"] for i in chosen} == {0, 1}
    assert set(diagnostics["unconstrained_selections"]) == {"bids", "less_task_max"}


def test_directional_instance_losses_recover_symmetric_objective():
    torch.manual_seed(7)
    output = (torch.nn.functional.normalize(torch.randn(5, 3), dim=-1),
              torch.nn.functional.normalize(torch.randn(5, 3), dim=-1), torch.tensor(3.))
    losses = directional_losses(output)
    torch.testing.assert_close((losses["text_to_image"].mean() + losses["image_to_text"].mean()) / 2, native_loss(output))
