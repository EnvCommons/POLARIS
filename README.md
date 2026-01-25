# POLARIS Environment

OpenReward environment for evaluating mathematical reasoning using the POLARIS-Project dataset.

## Overview

**Dataset:** POLARIS-Project/Polaris-Dataset-53K
**Tasks:** 53,291 math reasoning problems
**Type:** Single-turn answer verification
**Difficulty Levels:** 8 levels (0/8 to 7/8) based on pass rate estimation

## Features

- Mathematical answer verification using symbolic comparison
- Handles equivalent expressions (e.g., "1/2" equals "0.5")
- Difficulty tracking from easy (0/8) to hard (7/8) problems
- Single tool interface for clean evaluation

## Installation

### Prerequisites

- Python 3.11+
- OpenReward SDK
- OpenAI API key (for testing)

### Local Setup

```bash
# Clone the repository
cd polaris

# Install dependencies
pip install -r requirements.txt

# Download and prepare dataset (one-time setup)
python scripts/download_dataset.py

# Start local server
python server.py
```

The server will start on `http://localhost:8080`.

### Docker

```bash
# Build image
docker build -t polaris:latest .

# Run container
docker run -p 8080:8080 polaris:latest
```

## Usage

### Testing with Agent

```bash
# Set OpenAI API key
export OPENAI_API_KEY="your-key-here"

# Test single task
python test_agent.py

# Test multiple tasks
python test_agent.py multi 10
```

### Running Golden Tests

```bash
# Test 200 random tasks with correct and wrong answers (400 tests total)
pytest golden_tests.py -v
```

### Environment Structure

```python
from openreward import OpenReward

client = OpenReward()
environment = client.environments.get(name="EnvCommons/polaris")

# Get tasks
tasks = await environment.list_tasks(split="train")

# Start session
async with environment.session(task=tasks[0]) as session:
    prompt = await session.get_prompt()
    result = await session.call_tool("answer", {"answer": "42"})
    print(f"Reward: {result.reward}")
```

## Dataset Structure

Each task contains:
- `id`: Unique task identifier (0-53290)
- `problem`: The math problem statement
- `answer`: The expected answer
- `difficulty`: Estimated difficulty level (0/8 to 7/8)

Example task:
```json
{
  "id": "0",
  "problem": "The front tires of a car wear out after 25,000 km, and the rear tires wear out after 15,000 km. When should the tires be swapped so that they wear out at the same time?",
  "answer": "9375",
  "difficulty": "7/8"
}
```

## Tool: `answer`

Submit your final answer for evaluation.

**Parameters:**
- `answer` (string): Your final answer (number or mathematical expression)

**Returns:**
- `reward`: 1.0 if correct, 0.0 if incorrect
- `finished`: Always `true` (single-turn)
- `metadata`: Task ID, submitted answer, expected answer, correctness, difficulty

**Example:**
```python
result = await session.call_tool("answer", {"answer": "9375"})
# Returns: ToolOutput(reward=1.0, finished=True, blocks=[TextBlock(text="Correct!")])
```

## Data Preparation

The dataset is prepared using a one-time download script:

```bash
python scripts/download_dataset.py
```

This creates `data/polaris_tasks.parquet` (~11 MB) containing all 53,291 tasks.

**Note:** For deployment, this file is included in the Docker image. For development, run the download script once before starting the server.

## Development

### File Structure

```
polaris/
├── README.md                    # This file
├── polaris.py                   # Main environment class
├── server.py                    # Server wrapper
├── test_agent.py                # Agent integration tests
├── golden_tests.py              # Correctness tests
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container definition
├── scripts/
│   └── download_dataset.py      # Dataset preparation
└── data/
    └── polaris_tasks.parquet    # Task dataset (generated)
```

### Adding New Features

1. **Custom Verification Logic:** Modify the `answer()` tool in `polaris.py`
2. **Additional Splits:** Update `list_splits()` and `list_tasks()` methods
3. **Metadata:** Extend `PolarisTaskSpec` and `metadata` in ToolOutput

### Testing Checklist

- [ ] Syntax check: `python -m py_compile polaris.py server.py`
- [ ] Golden tests: `pytest golden_tests.py -v`
- [ ] Local server: `python server.py`
- [ ] Agent test: `python test_agent.py`
- [ ] Docker build: `docker build -t polaris:test .`
- [ ] Docker run: `docker run -p 8080:8080 polaris:test`

## Deployment

This environment is designed for deployment to OpenReward Runtime Service (ORS).

### Prerequisites

1. Create namespace at https://openreward.ai
2. Push code to GitHub repository
3. Configure deployment through OpenReward dashboard

### Deployment Steps

```bash
# Initialize git repository
git init
git add .
git commit -m "Initial POLARIS environment"

# Push to EnvCommons (requires GitHub token)
source /Users/rosstaylor/Documents/or_envs/newenvs/.env
gh repo create EnvCommons/polaris --public --source=. --remote=origin
git branch -M main
git push -u origin main
```

Then configure deployment at https://openreward.ai/environments/new.

## License

Apache 2.0 (same as POLARIS dataset)

## Citation

If you use this environment, please cite the original POLARIS dataset:

```bibtex
@dataset{polaris_dataset_53k,
  title={POLARIS-Dataset-53K},
  author={POLARIS Project},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/POLARIS-Project/Polaris-Dataset-53K}
}
```

## Support

For issues or questions:
- Environment issues: Open issue on GitHub
- Dataset questions: See [POLARIS dataset page](https://huggingface.co/datasets/POLARIS-Project/Polaris-Dataset-53K)
- OpenReward questions: See [OpenReward docs](https://docs.openreward.org/)
