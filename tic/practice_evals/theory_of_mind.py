"""To run this script:
  inspect eval theory_of_mind.py --model openai/gpt-3.5-turbo --max-connections 20;
  inspect view
"""

from inspect_ai import Task, eval, task
from inspect_ai.scorer import includes
from inspect_ai.solver import chain_of_thought, generate, self_critique, system_message
from inspect_ai.dataset import json_dataset


@task
def theory_of_mind():
    return Task(
        dataset=json_dataset("../input_files/practice_input_file.jsonl"),
        plan=[system_message("system.txt"), generate()],
        scorer=includes(),
    )
