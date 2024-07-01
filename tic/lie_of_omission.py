# %%
"""To run this script:
  inspect eval theory_of_mind.py --model openai/gpt-3.5-turbo --max-connections 20;
  inspect view
"""

from functools import partial

from inspect_ai import Task, eval, task
from inspect_ai.dataset import json_dataset
from inspect_ai.model import get_model
from inspect_ai.scorer import (
    accuracy,
    includes,
    bootstrap_std,
    Metric,
    metric,
    Scorer,
    scorer,
    Score,
    Target,
)
from inspect_ai.solver import generate, system_message, TaskState
import numpy as np


# %%
def lie_of_omission(
    dataset_json="",
    system_prompt_file="",
    scorerFn=includes(),
    metrics=[accuracy, bootstrap_std],
):

    @task
    def generateTask():
        # Generate is just the default plan of "send message to model and get response"
        # Add cache so we aren't reusing things.
        plan = [generate(cache=True)]
        if len(system_prompt_file) > 0:
            # Prepend system prompt
            plan = [system_message(system_prompt_file)] + plan

        return Task(
            dataset=json_dataset(dataset_json),
            plan=plan,
            # TODO: Have scorer that checks *not* includes
            # Or just do 1-p with two runs.
            scorer=scorerFn,
            # TODO: Make custom metric to handle not deception and what not.
            metrics=metrics,
        )

    return generateTask


# %%
# Scorer that can handle not including.
# TODO: Just make it a regex
@scorer(metrics=[accuracy(), bootstrap_std()])
def includesOrNot(ignore_case: bool = True, not_token="~"):

    async def score(state: TaskState, target: Target):

        # check for correct
        answer = state.output.completion
        target = target.text
        inverse_correct = False
        # handle and strip not token
        if target[0] == not_token:
            inverse_correct = True  # Not token
            target = target[1:]
        if ignore_case:
            correct = answer.lower().rfind(target.lower()) != -1
        else:
            correct = answer.rfind(target) != -1

        if inverse_correct:
            correct = not correct

        # return score
        return Score(value=1.0 if correct else 0.0, answer=answer)

    return score


# %%


# TODO: Make TP/FP/Precision/Recall metric
def var() -> Metric:
    """Compute variance over all scores."""

    def metric(scores: list[Score]) -> float:
        return np.var([score.as_float() for score in scores]).item()

    return metric


# %%

basic_deception_dataset_json = "input_files/basic_deception_detection.jsonl"
gpt4_generated_lie_of_omission_dataset_json = "input_files/qa_pairs_dataset.jsonl"
gpt4_generated_outright_lie_dataset_json = "input_files/truth_vs_lie_dataset.jsonl"
claude_generated_lie_system_prompt_v0 = (
    "tic_system_prompts/claude_generated_lie_system_prompt_v0.txt"
)
claude_generated_system_prompt_v0 = (
    "tic_system_prompts/claude_generated_system_prompt_v0.txt"
)


def basic_lie_of_omission():
    return lie_of_omission(
        dataset_json="",
        system_prompt_file="",
        scorerFn=includes(),
        metrics=[accuracy, bootstrap_std],
    )


eval(
    lie_of_omission(
        dataset_json=gpt4_generated_outright_lie_dataset_json,
        system_prompt_file=claude_generated_lie_system_prompt_v0,
        scorerFn=includesOrNot(),
    ),
    model="openai/gpt-4o",
    max_connections=20,
)

# %%

# TODO: Run async on all these models eventually
# https://ukgovernmentbeis.github.io/inspect_ai/eval-tuning.html#model-requests

models = [
    get_model("openai/gpt-3.5-turbo"),
    get_model("anthropic/claude-2.1"),
    #   get_model("mistral/mistral-large-latest")
]

# output = "Output to be scored"
# prompt = f"Could you please score the following output?\n\n{output}"

# graders = [model.generate(prompt) for model in models]

# grader_outputs = await asyncio.gather(*graders)

# %%
