# tests/test_data_hellaswag.py
import pytest

import llm_eval.data  # noqa: F401  (registers tasks)
from llm_eval.data.dataset import TASKS, Example, TaskType, get_task
from llm_eval.data.hellaswag import HellaSwagTask, _preprocess

ROWS = [
    {
        "ind": 24,
        "activity_label": "Roof shingle removal",
        "ctx_a": "A man is sitting on a roof.",
        "ctx_b": "he",
        "ctx": "A man is sitting on a roof. he",
        "endings": [
            "is using wrap to wrap a pair of skis.",
            "is ripping level tiles off.",
            "is holding a rubik's cube.",
            "starts pulling up roofing on a roof.",
        ],
        "source_id": "activitynet~v_-1IBHYS3L-Y",
        "split": "val",
        "split_type": "indomain",
        "label": "3",
    },
    {
        "ind": 180,
        "activity_label": "Preparing pasta",
        "ctx_a": "[header] How to cook pasta [title] Boil water.",
        "ctx_b": "add salt",
        "ctx": "",
        "endings": ["[substeps] Stir the pot.", "Drain [title] Serve.", "Wait  for it.", "Eat"],
        "source_id": "wikihow~299",
        "split": "val",
        "split_type": "zeroshot",
        "label": "0",
    },
    {
        "ind": 180,
        "activity_label": "Playing guitar",
        "ctx_a": "A woman holds a guitar.",
        "ctx_b": "she",
        "ctx": "",
        "endings": ["strums it.", "eats it.", "throws it.", "paints it."],
        "source_id": "activitynet~v_0bosp4-pyTM",
        "split": "val",
        "split_type": "indomain",
        "label": "1",
    },
    {
        "ind": 5,
        "activity_label": "Skiing",
        "ctx_a": "Snow falls.",
        "ctx_b": "a skier",
        "ctx": "",
        "endings": ["turns left.", "sleeps.", "swims.", "flies."],
        "source_id": "activitynet~v_x",
        "split": "val",
        "split_type": "indomain",
        "label": "0",
    },
    {
        "ind": 6,
        "activity_label": "Baking",
        "ctx_a": "Flour is on the table.",
        "ctx_b": "a cook",
        "ctx": "",
        "endings": ["kneads dough.", "drives away.", "sings.", "sleeps."],
        "source_id": "activitynet~v_y",
        "split": "val",
        "split_type": "indomain",
        "label": "0",
    },
    {
        "ind": 7,
        "activity_label": "Running",
        "ctx_a": "A track is empty.",
        "ctx_b": "a runner",
        "ctx": "",
        "endings": ["starts a lap.", "reads a book.", "cooks rice.", "paints."],
        "source_id": "activitynet~v_z",
        "split": "val",
        "split_type": "indomain",
        "label": "0",
    },
]


def make_task() -> HellaSwagTask:
    return HellaSwagTask(rows=lambda split: ROWS)


def test_preprocess_matches_lm_eval_rules():
    assert (
        _preprocess("[header] How to cook pasta [title] Boil water.")
        == " How to cook pasta. Boil water."
    )
    assert _preprocess("[substeps] Stir the pot.") == " Stir the pot."
    assert _preprocess("Wait  for it.") == "Wait for it."
    assert _preprocess("  Eat  ") == "Eat"


def test_load_builds_examples_with_row_position_ids_and_preprocessed_fields():
    examples = make_task().load("validation", None, seed=0)
    assert [e.id for e in examples] == [f"hellaswag/validation/{i}" for i in range(6)]
    first = examples[0]
    assert isinstance(first, Example)
    assert first.inputs["context"] == "Roof shingle removal: A man is sitting on a roof. He"
    assert first.choices == (
        "is using wrap to wrap a pair of skis.",
        "is ripping level tiles off.",
        "is holding a rubik's cube.",
        "starts pulling up roofing on a roof.",
    )
    assert first.target == "3"
    assert first.inputs["ind"] == 24 and first.inputs["source_id"] == "activitynet~v_-1IBHYS3L-Y"
    pasta = examples[1]
    assert pasta.inputs["context"] == "Preparing pasta: How to cook pasta. Boil water. Add salt"
    assert pasta.choices == (" Stir the pot.", "Drain. Serve.", "Wait for it.", "Eat")


def test_duplicate_native_ind_does_not_collide():
    examples = make_task().load("validation", None, seed=0)
    assert examples[1].inputs["ind"] == examples[2].inputs["ind"] == 180
    assert examples[1].id != examples[2].id


def test_context_renders_inputs_context():
    task = make_task()
    ex = task.load("validation", None, seed=0)[0]
    assert task.context(ex) == "Roof shingle removal: A man is sitting on a roof. He"


def test_subsets_are_deterministic_nested_and_int_indexed():
    task = make_task()
    two = [e.id for e in task.load("validation", 2, seed=7)]
    two_again = [e.id for e in task.load("validation", 2, seed=7)]
    four = [e.id for e in task.load("validation", 4, seed=7)]
    assert two == two_again
    assert set(two) <= set(four)
    assert four == sorted(four, key=lambda s: int(s.rsplit("/", 1)[1]))


def test_subset_matches_permutation_rule():
    from llm_eval.seeding import child_rng

    task = make_task()
    perm = child_rng(7, "subset", "hellaswag").permutation(6)
    expected = sorted(int(i) for i in perm[:3])
    assert [int(e.id.rsplit("/", 1)[1]) for e in task.load("validation", 3, seed=7)] == expected


def test_load_raises_when_n_exceeds_rows():
    with pytest.raises(ValueError, match="n_examples=7 exceeds 6 rows"):
        make_task().load("validation", 7, seed=0)


def test_dataset_info_after_load():
    task = make_task()
    with pytest.raises(RuntimeError):
        task.dataset_info()
    task.load("validation", 2, seed=0)
    assert task.dataset_info() == {
        "path": "Rowan/hellaswag",
        "revision": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
        "split": "validation",
        "n_rows": 6,
    }


def test_is_correct_compares_canonical_target():
    task = make_task()
    ex = task.load("validation", None, seed=0)[0]
    assert task.is_correct("3", ex) is True
    assert task.is_correct("1", ex) is False
    assert task.is_correct(None, ex) is False


def test_registry_exposes_hellaswag_and_rejects_unknown():
    assert TASKS["hellaswag"] is HellaSwagTask
    assert HellaSwagTask.task_type is TaskType.MULTIPLE_CHOICE
    assert isinstance(get_task("hellaswag"), HellaSwagTask)
    with pytest.raises(KeyError, match="unknown task 'nope'"):
        get_task("nope")


def test_target_text_is_the_gold_ending_and_letters_parse_back():
    task = make_task()
    ex = task.load("validation", None, seed=0)[0]
    assert task.target_text(ex) == "starts pulling up roofing on a roof."
    assert [task.extract_answer(g) for g in (" B", "(C)", "**D**", "so the answer is A.")] == [
        "1",
        "2",
        "3",
        "0",
    ]
    assert task.extract_answer(" A man walks in") is None
    assert task.extract_answer("no letter here") is None


def _pool_rows(n):
    return [dict(ROWS[2], ind=i, ctx_a=f"Event {i}.") for i in range(n)]


def test_exemplars_are_seeded_per_example_from_the_train_split():
    task = HellaSwagTask(rows=lambda split: _pool_rows(50))
    a = task.exemplars(3, 7, "hellaswag/validation/0")
    assert [e.id.split("/")[1] for e in a] == ["train"] * 3
    assert len({e.id for e in a}) == 3
    assert a == task.exemplars(3, 7, "hellaswag/validation/0")
    assert a != task.exemplars(3, 8, "hellaswag/validation/0")
    draws = {
        tuple(e.id for e in task.exemplars(3, 7, f"hellaswag/validation/{i}")) for i in range(5)
    }
    assert len(draws) > 1  # independent across examples
    for i in range(50):  # an example evaluated on train never sees itself
        assert f"hellaswag/train/{i}" not in [
            e.id for e in task.exemplars(3, 7, f"hellaswag/train/{i}")
        ]
