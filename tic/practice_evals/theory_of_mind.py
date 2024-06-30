"""To run this script:
  inspect eval theory_of_mind.py --model openai/gpt-3.5-turbo --max-connections 20;
  inspect view
"""

from inspect_ai import Task, eval, task
from inspect_ai.scorer import includes
from inspect_ai.solver import chain_of_thought, generate, self_critique, system_message
from inspect_ai.dataset import json_dataset
from inspect_ai.common import match_str, str_match_scorer
from inspect_ai.metrics import accuracy, bootstrap_std
from inspect_ai.scorer import Scorer, scorer, Score


# %%
@task
def theory_of_mind():
    return Task(
        dataset=json_dataset("../input_files/practice_input_file.jsonl"),
        plan=[system_message("system.txt"), generate(cache=True)],
        # TODO: Have scorer that checks *not* includes
        # Or just do 1-p with two runs.
        scorer=includesOrNot(),
        # TODO: Make custom metric to handle not deception and what not. 
        metrics=[custom_metric()]
    )


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
          inverse_correct = True # Not token
          target = target[1:]
        if ignore_case:
            correct = answer.lower().rfind(target.lower()) != -1
        else:
            correct = answer.rfind(target) != -1
        
        if inverse_correct:
            correct = not correct

        # return score
        return Score(
            value = 1.0 if correct else 0.0,
            answer=answer
        )

    return score

# %%
import numpy as np

from inspect_ai.scorer import Metric, Score, metric

# TODO: Make TP/FP/Precision/Recall metric
def var() -> Metric:
    """Compute variance over all scores."""

    def metric(scores: list[Score]) -> float:
        return np.var([score.as_float() for score in scores]).item()

    return metric
# %%

eval(theory_of_mind, model="openai/gpt-3.5-turbo")

# %%

# TODO: Run async on all these models eventually
# # https://ukgovernmentbeis.github.io/inspect_ai/eval-tuning.html#model-requests
# from inspect_ai.model import get_model

# models = [
#   get_model("openai/gpt-4"),
#   get_model("anthropic/claude-3-sonnet-20240229"),
#   get_model("mistral/mistral-large-latest")
# ]

# output = "Output to be scored"
# prompt = f"Could you please score the following output?\n\n{output}"

# graders = [model.generate(prompt) for model in models]

# grader_outputs = await asyncio.gather(*graders)
