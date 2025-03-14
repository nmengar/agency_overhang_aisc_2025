from langchain_chroma import Chroma
from langchain_core.documents import Document
from inspect_ai import Task, task
from inspect_ai.log import read_eval_log
from inspect_ai.dataset import example_dataset
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import (
    chain_of_thought, generate, self_critique, solver, chain
)
from uuid import uuid4
from inspect_ai import eval
from benchmark_tasks.mbpp import MBPP
from benchmark_tasks.hellaswag import HellaSwagTask


def initialize_database(collection_name, embedding_fn, persist_directory):
    vector_store = Chroma(
        collection_name = collection_name,
        embedding_function = embedding_fn,
        persist_directory = persist_directory
    )
    return vector_store

@solver
def few_shot_prompts():
    return chain(
        generate()
    )

@task
def theory_of_mind():
    return Task(
        dataset = example_dataset("theory_of_mind"),
        solver = [
            chain_of_thought(),
            generate(),
            self_critique()
        ],
        scorer=model_graded_fact()
    )

if __name__ == '__main__':
    eval_logs = eval(tasks=HellaSwagTask, model='openai/gpt-4o', log_dir='./logs', log_format='json', score=True, score_display=False)
    read_eval_log(
        log_file=eval_logs[0],
        format='json')
