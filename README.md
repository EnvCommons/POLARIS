# POLARIS

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/EnvCommons/polaris) [![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/POLARIS-Project/Polaris-Dataset-53K)

## Description

POLARIS is an environment for evaluating mathematical reasoning capabilities. It contains 53,291 math reasoning problems with 8 difficulty levels (0/8 to 7/8) based on pass rate estimation. The environment uses symbolic comparison for answer verification, handling equivalent expressions.

## Capabilities

- Mathematical reasoning across difficulty levels
- Symbolic answer verification
- Equivalent expression handling
- Multi-format numerical answers

## Compute Requirements

Agents are given a standard environment with no sandbox or file system access.

## License

[Apache 2.0](https://opensource.org/licenses/Apache-2.0).

## Tasks

There is one split in this environment:

- **train**: 53,291 tasks

Tasks span 8 difficulty levels from easy (0/8) to hard (7/8).

## Reward Structure

This is a single-turn environment. The agent submits an answer via the `answer` tool. Verification uses symbolic comparison, handling equivalent expressions (e.g., "1/2" equals "0.5"). Reward is binary: 1.0 if correct, 0.0 if incorrect.

## Data

Data consists of a Parquet file (`polaris_tasks.parquet`, ~11 MB) sourced from [HuggingFace POLARIS-Project/Polaris-Dataset-53K](https://huggingface.co/datasets/POLARIS-Project/Polaris-Dataset-53K). Each row contains a problem statement, expected answer, and difficulty level. Data is stored on the OpenReward platform.

## Tools

| Tool | Description |
|------|-------------|
| `answer` | Submit your final answer (number or expression). Ends the episode. |

## Time Horizon

Single-turn. The agent reads the math problem and submits one answer.

## Environment Difficulty

POLARIS evaluates mathematical reasoning across 8 difficulty levels with symbolic answer verification.

## Other Environment Requirements

None. All evaluation is deterministic using symbolic comparison.

## Safety

Agents in POLARIS solve mathematical problems in a standard environment. The environment does not present direct safety risks.

## Citation

```bibtex
@dataset{polaris_dataset_53k,
  title={POLARIS-Dataset-53K},
  author={POLARIS Project},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/POLARIS-Project/Polaris-Dataset-53K}
}
```
