import torch
from vllm import SamplingParams, LLM

examples = [
    {
        "prompt": 'A conversation between user and assistant. The user asks a question, and the assistant solves it. The time limit is set to 20,480 tokens. If the assistant\'s response exceeds this limit, a progressively increasing penalty with the number of tokens exceeded will be applied.\nuser\nSolve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\nAmong the 900 residents of Aimeville, there are 195 who own a diamond ring, 367 who own a set of golf clubs, and 562 who own a garden spade. In addition, each of the 900 residents owns a bag of candy hearts. There are 437 residents who own exactly two of these things, and 234 residents who own exactly three of these things. Find the number of residents of Aimeville who own all four of these things.\nRemember to put your answer on its own line after "Answer:".\nassistant',
        "answer": "73",
    },
    {
        "prompt": 'A conversation between user and assistant. The user asks a question, and the assistant solves it. The time limit is set to 20,480 tokens. If the assistant\'s response exceeds this limit, a progressively increasing penalty with the number of tokens exceeded will be applied.\nuser\nSolve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\nConsider the paths of length $16$ that follow the lines from the lower left corner to the upper right corner on an $8\times 8$ grid. Find the number of such paths that change direction exactly four times, as in the examples shown below.\nRemember to put your answer on its own line after "Answer:".\nassistant',
        "answer": "294",
    },
    {
        "prompt": 'A conversation between user and assistant. The user asks a question, and the assistant solves it. The time limit is set to 20,480 tokens. If the assistant\'s response exceeds this limit, a progressively increasing penalty with the number of tokens exceeded will be applied.\nuser\nSolve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\nA list of positive integers has the following properties:\n$\\bullet$ The sum of the items in the list is $30$.\n$\\bullet$ The unique mode of the list is $9$.\n$\\bullet$ The median of the list is a positive integer that does not appear in the list itself.\nFind the sum of the squares of all the items in the list.\nRemember to put your answer on its own line after "Answer:".\nassistant',
        "answer": "236",
    },
]


def main():
    model = "./DAPO-Qwen-32B"

    llm = LLM(
        model=model,
        dtype=torch.bfloat16,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.95,
    )

    sampling_params = SamplingParams(temperature=1.0, top_p=0.7, max_tokens=32768)

    for example in examples:
        prompt = example["prompt"]
        answer = example["answer"]
        output = llm.generate(prompt, sampling_params)
        print(
            f"***QUESTION***:\n{prompt}\n***GROUND TRUTH***:\n{answer}\n***MODEL OUTPUT***:\n{output[0].outputs[0].text}\n"
        )
        print("-" * 100)


if __name__ == "__main__":
    main()
