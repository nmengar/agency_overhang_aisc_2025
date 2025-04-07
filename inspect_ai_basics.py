from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate, multiple_choice, solver, TaskState, Generate, system_message
from inspect_ai.scorer import choice, exact
from inspect_ai import eval
from inspect_ai.model import GenerateConfig, Model, get_model
from inspect_ai._util.dict import omit
from inspect_ai._util.format import format_template
from inspect_ai.util import resource
import json

@solver
def custom_solver(template: str):
    with open(template,'r') as f:
        prompt_template = resource(f.read())
    async def solve(state: TaskState, generate: Generate):
        prompt = state.user_prompt
        kwargs = omit(state.metadata | state.store._data, ["prompt"])
        prompt.text = format_template(prompt_template, {"prompt": prompt.text} | kwargs)
        return state
    return solve

@solver
def custom_solver_1(template: str, optimizer_model: Model, store_prompts: bool, prompt_db: dict):
    with open(template,'r') as f:
        prompt_template = resource(f.read())
    async def solve(state: TaskState, generate: Generate):
        nonlocal optimizer_model
        prompt = state.user_prompt
        kwargs = omit(state.metadata | state.store._data, ["prompt"])
        prompt_optimizer_input = format_template(prompt_template, {"prompt": prompt} | kwargs)
        response = await optimizer_model.generate(input=prompt_optimizer_input, config=GenerateConfig(max_retries=2,timeout=5))
        # if store_prompts:
        #     prompt_database = {}
        #     with open('./enhanced_prompts/prompt_db.json','r+') as f:
        #         prompt_database = json.load(f)
        #     with open('./enhanced_prompts/prompt_db.json','w+') as f:
        #         prompt_database[prompt_db['benchmark']][prompt_db['base_model']][prompt_optimizer_input][prompt_db['prompt_optimizer_model']] = response.completion
        #         json.dump(prompt_database)
        prompt.text = format_template(response.choices[0].message.content, {"prompt": prompt.text} | kwargs)
        return state
    return solve

@task
def hello_world():
    return Task(
        dataset=[
            Sample(
                input="Just say 'Hello World'",
                target="Hello World LOL",
            ),
        ],
        solver=[
            custom_solver('./prompt_templates/hello_world.txt'),
            generate(),
        ],
        scorer=exact(),
    )

@task
def hello_world_1():
    return Task(
        dataset=[
            Sample(
                input="A group of engineers wanted to know how different building designs would respond during an earthquake. They made several models of buildings and tested each for its ability to withstand earthquake conditions. Which will most likely result from testing different building designs?",
                choices=[
                    "buildings will be built faster",
                    "buildings will be made safer",
                    "building designs will look nicer",
                    "building materials will be cheaper"
                ],
                target="B",
            ),
        ],
        scorer=choice(),
    )

if __name__ == '__main__':

    # Running eval for a task
    # eval(
    #     tasks=hello_world,
    #     model="google/gemini-1.5-pro",
    #     log_dir="./logs/",
    #     log_format='json',
    # )

    # with open('./prompt_templates/generated_knowledge.txt','r') as f:
    #     print(f.read())
    # prompt_template = resource('./prompt_templates/generated_knowledge.txt')
    # print(prompt_template)
    opt_model = get_model(model='openai/gpt-4o-mini',config=GenerateConfig(max_retries=2,timeout=5),memoize=False)
    eval(
        tasks=Task(
            dataset=[
                Sample(
                    input="A group of engineers wanted to know how different building designs would respond during an earthquake. They made several models of buildings and tested each for its ability to withstand earthquake conditions. Which will most likely result from testing different building designs?",
                    choices=[
                        "buildings will be built faster",
                        "buildings will be made safer",
                        "building designs will look nicer",
                        "building materials will be cheaper"
                    ],
                    target="B",
                )
            ],
            solver=[
                custom_solver_1(
                    template='./prompt_templates/generated_knowledge.txt',optimizer_model=opt_model,
                    store_prompts = True,
                    prompt_db = {
                        "benchmark": "hello_world",
                        "base_model": "groq/gemma2-9b-it",
                        "prompt_optimizer_model": "openai/gpt-4o-mini",
                    }
                ),
                multiple_choice(),
            ],
        ),
        task_args={
            "scorer": choice()
        },
        model='groq/gemma2-9b-it',
        log_dir="./logs/",
        log_format='json',
    )