# %%
import os
import random

import openai
import orjson
import utils
from tqdm import tqdm

# %%
if openai.api_key is None:
    with open("./openai-api-key") as f:
        openai.api_key = f.read().strip()
# print(openai.api_key)

query_set_name = "prm800k-002validation-seed42"


# %%
def prm800k_query_openai_to_analyse_and_rate_sample(
    sample, verbose=False, test=False, debug=False
):
    reformatted_sample = utils.reformat_prm800k_sample(sample)
    task_prefix = "# Solution to analyse every step and rate"
    problem_prefix = "## Problem"
    problem = reformatted_sample["problem"]
    steps = [item["step"] for item in reformatted_sample["step_ratings"]]
    steps_prefix = "## Steps"
    steps_prompt = "\n".join(
        [f'Step {i+1}: """{step}"""' for i, step in enumerate(steps)]
    )
    step_analysis_rating_prefix = "## Step-Analysis-Rating"
    step_analysis_rating_example = f'Step 1: """{steps[0]} """ Analysis: this step '
    task_prompt = f"{task_prefix}\n{problem_prefix}\n{problem}\n{steps_prefix}\n{steps_prompt}\n{step_analysis_rating_prefix}\n{step_analysis_rating_example}"

    # print(task_prompt)
    sys_message = "# Instructions\nYou're an excellent mathematician who excels at critically analyze the steps in the solutions to math problems and find errors in them.\nI will provide some math problem and its step-by-step solution, and your goal is to first critically analyze every step and rate this to math problems with negative(-1) or neutral(0) or positive(1).\nIf a step is not compliant with any one among the following constraints: \n- Appropriate in conversation \n- Contains no inaccuracies \n- Contains no weirdness \n- Computations can be verified with recomputation or other methods\nthen this step should be rated as -1.\nIf a step is compliant with all of the constraints above, then this step can be rated as 0.\nAnd if this step additionally advances the process of solving the problem, then this step should be rated as 1 instead of 0.\nThe solutions will often say things that look ok at first, but will turn out to be wrong on closer inspection - stay vigilant!\nNever make any conlusion such a rating or whether this step is correct before your analysis finishes.\nYou must analyze and rate every step until the solution finishes.\nThe format should be: Step {i}: {step} Analysis: this step {analysis} Rating: {rating}\nLet's take a deep breath first and think step by step.\n# Example\n## Problem\nLet $\\theta$ be the smallest acute angle for which $\\sin \\theta,$ $\\sin 2 \\theta,$ $\\sin 3 \\theta$ form an arithmetic progression, in some order.  Find $\\cos \\theta.$\n## Step-Analysis-Rating\nStep 1: I notice that the problem involves trigonometric functions and arithmetic progressions, so I wonder if there is a connection between them. Analysis: this step points out what the problem involves and leads to probing into the connection between them, so this step is appropriate in conversation, contains no inaccuracies, contains no weirdness, and contains no computations to verify, but fails to substantially advance the process of solving the problem, so it should be rated as 0. Rating: 0\nStep 2: I recall that an arithmetic progression is a sequence of numbers where each term is obtained by adding a constant amount to the previous term. Analysis: this step recalls the definition of an arithmetic progression, which is accurate and relevant to the problem. It is also appropriate in conversation and contains no weirdness. The computation is not applicable here, but the information provided helps to advance the solution process, so it should be rated as 1. Rating: 1\nStep 3: I also remember that the sine function is periodic, which means that it repeats the same values over and over again at regular intervals. Analysis: this step recalls an important characteristic of the sine function, which is accurate and directly relevant to the problem. It's also appropriate in conversation and contains no weirdness. There's no computation to verify in this step, but it advances the solution process, so it should be rated as 1. Rating: 1\nStep 4: I wonder if I can use these facts to find a relationship between $\\sin \\theta,$ $\\sin 2 \\theta,$ and $\\sin 3 \\theta.$ Analysis: this step is a reflection on the previously mentioned facts and poses a question about how to apply them to solve the problem. It's appropriate in conversation, contains no inaccuracies or weirdness. However, it does not involve any computations and does not advance the solution process, hence it's rated as 0. Rating: 0\nStep 5: I try to visualize what the graph of the sine function looks like, and how it changes when I multiply the argument by 2 or 3. Analysis: this step involves visualizing the problem, which is a valid and effective approach in mathematics. It's appropriate in conversation, contains no inaccuracies or weirdness, and while there's no computation, it advances the process of solving the problem by providing a way to understand the problem, so it should be rated as 1. Rating: 1\nStep 6: I notice that multiplying the argument by 2 makes the graph oscillate twice as fast, and multiplying by 3 makes it oscillate three times as fast. Analysis: this step provides an accurate observation about the behavior of the sine function when the argument is multiplied. It's appropriate in conversation, contains no weirdness, and while there's no computation, it does advance the solution process, so it should be rated as 1. Rating: 1\nStep 7: I also notice that the amplitude of the graph stays the same, which means that the highest and lowest values of the sine function are still 1 and -1, regardless of the argument. Analysis: this step provides another accurate observation about the sine function. It's appropriate in conversation, contains no weirdness, and while there's no computation, it does advance the solution process, so it should be rated as 1. Rating: 1\nStep 8: I think about what it means for three values of the sine function to form an arithmetic progression. Analysis: this step reflects on the problem statement and what it implies. It's appropriate in conversation, contains no inaccuracies or weirdness, and while there's no computation, it does not advance the solution process, hence it's rated as 0. Rating: 0\nStep 9: It means that the difference between any two consecutive values is the same. Analysis: this step provides an accurate definition of an arithmetic progression in the context of the problem. It's appropriate in conversation, contains no weirdness, and while there's no computation, it does advance the solution process, so it should be rated as 1. Rating: 1\nStep 10: For example, if $\\sin \\theta = a,$ $\\sin 2 \\theta = b,$ and $\\sin 3 \\theta = c,$ then $b - a = c - b,$ or $2b = a + c.$ Analysis: this step correctly applies the definition of an arithmetic progression to the problem. It's appropriate in conversation, contains no weirdness, and in the computation, for $b - a = c - b$, moving the term $b$ yields $2b = a + c$, which verifies the computation, so it advances the solution process and should be rated as 1. Rating: 1\nStep 11: I wonder if this equation has any solutions for $a, b, c$ between -1 and 1, which are the possible values of the sine function. Analysis: this step reflects on the possible solutions of the equation. It's appropriate in conversation, contains no inaccuracies or weirdness, and while there's no computation, it does not advance the solution process, hence it's rated as 0. Rating: 0\nStep 12: I try to simplify the equation by using some trigonometric identities. Analysis: this step proposes a valid approach to solve the equation. It's appropriate in conversation, contains no inaccuracies or weirdness, and while there's no computation, it does advance the solution process, so it should be rated as 1. Rating: 1\nStep 13: I know that $\\sin 2 \\theta = 2 \\sin \\theta \\cos \\theta,$ and $\\sin 3 \\theta = 3 \\sin \\theta - 4 \\sin^3 \\theta.$ Analysis: this step correctly recalls two important trigonometric identities. It's appropriate in conversation, contains no weirdness, and while there's no computation, it does advance the solution process, so it should be rated as 1. Rating: 1\nStep 14: I substitute these expressions into the equation and get $4 \\sin \\theta \\cos \\theta = \\sin \\theta + 3 \\sin \\theta - 4 \\sin^3 \\theta.$ Analysis: this step correctly substitutes the trigonometric identities into the equation. It's appropriate in conversation, contains no weirdness, and the calculations in this step substitute some trigonometric identities into the equation $2b = a + c$ (where $\\sin \\theta = a,$ $ $\\sin 2 \\theta = b,$ and $\\sin 3 \\theta = c$) to get $4 \\sin \\theta \\cos \\theta = \\sin \\theta + 3 \\sin \\theta - 4 \\sin^3 \\theta$. sin^3 \\theta$, and to verify these calculations, we note that $\\sin 2\\theta = 2\\sin \\theta \\cos \\theta, \\sin 3\\theta = 4\\sin \\theta - 4\\sin^3 \\theta$, and substituting does give the same result, so the calculations in this step can be verified. By this substition, this step advances the solution process and should be rated as 1. Rating: 1\nStep 15: I divide both sides by $\\sin \\theta$ and get $4 \\cos \\theta = 4 - 4 \\sin^2 \\theta.$ Analysis: this step correctly simplifies the equation by dividing both sides by $\\sin \\theta$. It's appropriate in conversation, contains no weirdness, and in the computation, both sides so it advances the solution process and should be rated as 1. Rating: 1\nStep 16: I recognize that $\\sin^2 \\theta = 1 - \\cos^2 \\theta,$ so I substitute that and get $4 \\cos \\theta = 4 - 4 (1 - \\cos^2 \\theta).$ Analysis: this step correctly applies the Pythagorean identity $\\sin^2 \\theta = 1 - \\cos^2 \\theta$. It's appropriate in conversation, contains no weirdness, and the computation is simple substitution that can be easily verified, so it advances the solution process and should be rated as 1. Rating: 1\nStep 17: I simplify and get $4 \\cos^2 \\theta + 4 \\cos \\theta - 4 = 0.$ Analysis: this step simplifies $4 \\cos \\theta = 4 - 4 (1 - \\cos^2 \\theta)$ to get $4 \\cos^2 \\theta + 4 \\cos \\theta - 4 = 0$, but by expanding the expression and distributing $-4$ on the right side of the equation, we get $4 \\cos \\theta = 4 + 4 \\cos^2 \\theta - 4$, then combining like terms on the right side of the equation, we get $4 \\cos \\theta = 4\\cos^2 \\theta$, and then isolating $\\cos \\theta$ on one side by dividing both sides by 4, we get $\\frac{4 \\cos \\theta}{4} = \\frac{4\\cos^2 \\theta}{4}$, which further simplifies to $\\cos \\theta = \\cos^2 \\theta$. Therefore, it only looks appropriate in conversation and without weirdness, but contains inaccuracies and the computation is wrong. Thus, it should be rated as -1. Rating: -1"

    request_dict = dict(
        model="gpt-4",
        messages=[
            {"role": "system", "content": sys_message},
            {
                "role": "user",
                "content": task_prompt,
            },
        ],
        temperature=0.6,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
    )

    if not test:
        response = openai.ChatCompletion.create(**request_dict)
    else:
        response = {
            "id": "chatcmpl-7xxu1TgC5ioae40MNWuW0q3NRZQHD",
            "object": "chat.completion",
            "created": 1694525973,
            "model": "gpt-4-0613",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "correctly identifies the groups that need to be arranged. It's appropriate in conversation, contains no inaccuracies or weirdness, and while there's no computation, it doesn't advance the solution process, so it should be rated as 0. Rating: 0\nStep 2: I wonder how many ways I can arrange these groups around the circle, ignoring the order within each group for now. Analysis: this step is a reflection on the problem and proposes a valid approach to solve it. It's appropriate in conversation, contains no inaccuracies or weirdness, and while there's no computation, it does advance the solution process, so it should be rated as 1. Rating: 1\nStep 3: I recall that the number of ways to arrange n distinct objects around a circle is (n-1)!, since we can fix one object and then permute the rest. Analysis: this step correctly recalls the formula for arranging objects around a circle, which is relevant to the problem. It's appropriate in conversation, contains no weirdness, and while there's no computation, it does advance the solution process, so it should be rated as 1. Rating: 1\nStep 4: So, for the three groups, there are (3-1)! = 2! = 2 ways to arrange them around the circle. Analysis: this step correctly applies the formula to the problem. It's appropriate in conversation, contains no weirdness, and the computation is correct and can be easily verified, so it advances the solution process and should be rated as 1. Rating: 1\nStep 5: For example, one way is D-R-I, and the other way is R-D-I, where D stands for Democrats, R for Republicans, and I for Independent. Analysis: this step provides valid examples of the possible arrangements of the groups. It's appropriate in conversation, contains no inaccuracies or weirdness, and while there's no computation, it does advance the solution process, so it should be rated as 1. Rating: 1\nStep 6: Now, I need to consider the order within each group. Analysis: this step correctly identifies the next step in the problem-solving process. It's appropriate in conversation, contains no inaccuracies or weirdness, and while there's no computation, it does advance the solution process, so it should be rated as 1. Rating: 1\nStep 7: For the Democrats, there are 5! ways to order them in a line, but only half of them are distinct around a circle, since reversing the order gives the same seating. Analysis: this step correctly applies the formula for arranging objects in a line to the Democrats and correctly notes that only half of the arrangements are distinct around a circle. It's appropriate in conversation, contains no weirdness, and the computation is correct and can be easily verified, so it advances the solution process and should be rated as 1. Rating: 1",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 2644,
                "completion_tokens": 609,
                "total_tokens": 3253,
            },
        }

    # query = response.copy()
    response["task_prompt"] = task_prompt

    if verbose:
        print(response)
    return response


def prm800k_query_openai_to_analyse_and_rate_dataset(
    dataset="prm800k-002validation-seed42",
    query_set_name: str = None,
    resume=True,
    restart=False,
    debug: bool = False,
):
    if isinstance(dataset, str):
        if not os.path.exists(dataset):
            dataset = os.path.join(utils.project_root, "datasets", f"{dataset}.jsonl")
    if dataset.endswith(".jsonl"):
        dataset = utils.load_jsonl(dataset)

    if debug:
        print(random.choice(dataset))

    queries_path = os.path.join(
        utils.project_root, "datasets", f"{query_set_name}-openai-api-queries.jsonl"
    )

    print(f"queries_path = {queries_path}")

    if restart:
        os.remove(queries_path)

    if not os.path.exists(queries_path):
        file_mode = "w"
    else:
        file_mode = "r+"

    with open(queries_path, file_mode) as f:
        # r+: read from the beginning and write to the end
        if resume:
            lines = f.readlines()
            num_completed = len(lines)
            if debug:
                print(lines)
                print(f"num_completed = {num_completed}")
                # raise RuntimeError()
        for idx, sample in tqdm(enumerate(dataset)):
            if idx < num_completed:
                continue
            query = prm800k_query_openai_to_analyse_and_rate_sample(sample, test=debug)
            f.write(orjson.dumps(query).decode() + "\n")
            if debug:
                break
    return None


# %%

if __name__ == "__main__":
    prm800k_query_openai_to_analyse_and_rate_dataset(
        dataset=query_set_name,
        query_set_name=query_set_name,
        resume=True,
        # restart=True,
        # debug=True,
    )
