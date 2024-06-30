from inspect_ai import Task, eval, task

# TODO: Remove
from inspect_ai.dataset import example_dataset
from inspect_ai.scorer import includes
from inspect_ai.solver import chain_of_thought, generate, self_critique
from inspect_ai.dataset import json_dataset


@task
def theory_of_mind():
    return Task(
        dataset=json_dataset("../input_files/practice_input_file_16.jsonl"),
        # plan=[chain_of_thought(), generate(), self_critique()],
        scorer=includes(),
    )
