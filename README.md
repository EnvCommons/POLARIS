# POLARIS

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://www.openreward.ai/GeneralReasoning/POLARIS-53K) [![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/POLARIS-Project/Polaris-Dataset-53K)

## Description

POLARIS is an environment for evaluating mathematical reasoning capabilities. It contains 53,291 math reasoning problems with 8 difficulty levels (0/8 to 7/8) based on pass rate estimation. The environment serves a repaired copy of the dataset: questions whose original key could not grade them correctly (garbled keys, option letters, multi-part questions, proofs stored with a fragment of the statement as key, leaked answers) were fixed, and the task order is unchanged.

## Capabilities

- Mathematical reasoning across difficulty levels
- Symbolic answer verification with equivalent expression handling
- Questions with several valid answers
- Proof writing, graded against a reference proof and rubric

## Compute Requirements

Agents are given a standard environment with no sandbox or file system access.

## License

[MIT](https://opensource.org/licenses/MIT).

## Tasks

There are two splits in this environment:

- **train**: 53,291 tasks
- **train_old**: 45,840 tasks, the subset of train whose problem text passes a pre-2000 knowledge cutoff filter

Tasks span 8 difficulty levels from easy (0/8) to hard (7/8).

## Reward Structure

This is a single-turn environment. The agent submits an answer via the `answer` tool, and only the first graded answer is rewarded; a repeat submission gets a reward of -0.1, and an empty answer is not graded and leaves the episode open.

Most tasks (49,674) are verified by symbolic comparison with math_verify, handling equivalent expressions (e.g., "1/2" equals "0.5"); reward is 1.0 if correct and 0.0 otherwise. The other 3,617 tasks are graded by an LLM judge (gpt-5-mini by default): 934 questions whose key math_verify cannot compare reliably (several valid answers, text answers, relations) get a correct/incorrect verdict (reward 1.0 or 0.0), and 2,683 proof tasks get a 0-7 rubric score against a reference proof (reward = score / 7). If the judge cannot be reached, the submission is returned without a reward instead of being scored 0.

## Data

Data consists of a Parquet file (`polaris_tasks.parquet`, ~27 MB) built from [HuggingFace POLARIS-Project/Polaris-Dataset-53K](https://huggingface.co/datasets/POLARIS-Project/Polaris-Dataset-53K). Each row contains a problem statement, the answer key and the difficulty level, plus the grading mode, the answer format, judge notes (accepted answers or a proof rubric) and, for the 6,605 repaired rows, the original problem and key. Data is stored on the OpenReward platform.

## Tools

| Tool | Description |
|------|-------------|
| `answer` | Submit your final answer (number, expression, list of answers, or a full proof for proof tasks). Ends the episode, unless the answer is empty, which is not graded. |

## Time Horizon

Single-turn. The agent reads the math problem and submits one answer.

## Environment Difficulty

[Put environment difficulty statistics here]

## Other Environment Requirements

The 3,617 judge-graded tasks need an OpenAI API key, passed as the `openai_api_key` secret. The judge follows `OPENAI_BASE_URL` when it is set, so it can be pointed at any OpenAI-compatible endpoint. Tasks verified by math_verify need no key.

## Safety

Agents in POLARIS solve mathematical problems in a standard environment. The environment does not present direct safety risks.

## Citation

```bibtex
@misc{Polaris2025,
  title={POLARIS: A Post-Training Recipe for Scaling Reinforcement Learning on Advanced Reasoning Models},
  url={https://hkunlp.github.io/blog/2025/Polaris},
  author={An, Chenxin and Xie, Zhihui and Li, Xiaonan and Li, Lei and Zhang, Jun and Gong, Shansan and Zhong, Ming and Xu, Jingjing and Qiu, Xipeng and Wang, Mingxuan and Kong, Lingpeng},
  year={2025}
}
```
