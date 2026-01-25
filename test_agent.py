"""
Test agent for POLARIS environment.
Demonstrates integration with OpenAI's Responses API.
"""
import asyncio
import json
import os

from openai import AsyncOpenAI
from openreward import AsyncOpenReward

# Configuration
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-5")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable must be set")


async def test_polaris_environment():
    """Test the POLARIS environment with a single task"""

    # Initialize clients
    or_client = AsyncOpenReward()
    oai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    # Get environment and tasks
    environment = or_client.environments.get(name="local/Polaris", base_url="http://localhost:8080")
    tasks = await environment.list_tasks(split="train")
    tools = await environment.list_tools(format="openai")

    print(f"Loaded {len(tasks)} tasks from POLARIS environment")

    # Test first task
    task = tasks[0]
    print(f"\n{'='*60}")
    print(task)
    print(f"{'='*60}\n")

    async with environment.session(task=task, secrets={}) as session:
        # Get initial prompt
        prompt = await session.get_prompt()
        input_list = [{"role": "user", "content": prompt[0].text}]

        finished = False
        turn = 0

        while not finished and turn < 10:  # Safety limit
            turn += 1
            print(f"\nTurn {turn}:")

            # Call model
            response = await oai_client.responses.create(
                model=MODEL_NAME,
                tools=tools,
                input=input_list,
            )

            # Process response
            input_list += response.output

            for item in response.output:
                if item.type == "function_call":
                    print(f"  Tool call: {item.name}")
                    print(f"  Arguments: {item.arguments}")

                    # Execute tool
                    tool_result = await session.call_tool(
                        item.name,
                        json.loads(str(item.arguments)),
                    )

                    # Extract results
                    reward = tool_result.reward
                    finished = tool_result.finished
                    feedback = tool_result.blocks[0].text if tool_result.blocks else ""

                    print(f"  Result: {feedback}")
                    print(f"  Reward: {reward}")
                    print(f"  Finished: {finished}")

                    # Add tool result to conversation
                    input_list.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": json.dumps({"result": feedback})
                    })

                    if finished:
                        print(f"\n{'='*60}")
                        print(f"Episode Complete!")
                        print(f"Final Reward: {reward}")
                        print(f"Metadata: {tool_result.metadata}")
                        print(f"{'='*60}")
                        break

                elif item.type == "text":
                    print(f"  Model: {item.text[:100]}...")

            # Check for non-tool-using finish
            if not any(i.type == "function_call" for i in response.output):
                print("  Model did not call any tools. Stopping.")
                break


async def test_multiple_tasks(num_tasks: int = 5):
    """Test multiple tasks to verify environment stability"""

    or_client = OpenReward(base_url="http://localhost:8080")
    oai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    environment = or_client.environments.get(name="local/Polaris")
    tasks = await environment.list_tasks(split="train")
    tools = await environment.list_tools(format="openai")

    print(f"\nTesting {num_tasks} tasks...")

    results = []
    for i, task in enumerate(tasks[:num_tasks]):

        async with environment.session(task=task, secrets={}) as session:
            prompt = await session.get_prompt()
            input_list = [{"role": "user", "content": prompt[0].text}]

            response = await oai_client.responses.create(
                model=MODEL_NAME,
                tools=tools,
                input=input_list,
            )

            for item in response.output:
                if item.type == "function_call":
                    tool_result = await session.call_tool(
                        item.name,
                        json.loads(str(item.arguments)),
                    )

                    results.append({
                        "task_id": task["id"],
                        "difficulty": task["difficulty"],
                        "reward": tool_result.reward,
                        "correct": tool_result.metadata.get("correct", False),
                    })

                    status = "✓" if tool_result.reward == 1.0 else "✗"
                    print(f"  {status} Reward: {tool_result.reward}")
                    break

    # Summary
    print(f"\n{'='*60}")
    print(f"Results Summary:")
    print(f"  Total tasks: {len(results)}")
    print(f"  Correct: {sum(r['correct'] for r in results)}")
    print(f"  Average reward: {sum(r['reward'] for r in results) / len(results):.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "multi":
        num_tasks = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        asyncio.run(test_multiple_tasks(num_tasks))
    else:
        asyncio.run(test_polaris_environment())
