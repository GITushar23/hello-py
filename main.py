import asyncio
import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolUnionParam

from task import PROMPT, TOOL_HANDLERS, TOOLS, grading_func, reset_task_state

MAX_TOKENS = 5000
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)


async def run_agent_loop(
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    max_steps: int = 20,
    model: str = "claude-haiku-4-5",
    verbose: bool = True,
    log_file: Path | None = None,
) -> Any | None:
    """Runs an agent loop with the given prompt and tools."""
    client = AsyncAnthropic()
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "max_steps": max_steps,
        "initial_prompt": prompt,
        "steps": [],
        "final_result": None,
        "completion_status": None,
    }

    for step in range(max_steps):
        if verbose:
            print(f"\n{'='*60}\n⚡ Step {step + 1}/{max_steps}\n{'='*60}")

        response = await client.messages.create(
            model=model, max_tokens=MAX_TOKENS, tools=tools, messages=messages
        )

        step_log = {
            "step_number": step + 1,
            "api_response": {
                "id": response.id,
                "stop_reason": response.stop_reason,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            },
            "tool_calls": [],
        }

        if response.stop_reason == "max_tokens":
            print(f"Model reached max_tokens limit {MAX_TOKENS}")

        has_tool_use = False
        tool_results = []
        submitted_answer = None

        for content in response.content:
            if content.type == "text":
                if verbose:
                    print(f"\n💭 Assistant: {content.text}")
                step_log["tool_calls"].append({"type": "text", "content": content.text})

            elif content.type == "tool_use":
                has_tool_use = True
                tool_name = content.name
                tool_input = content.input

                tool_call_log = {
                    "tool_name": tool_name,
                    "tool_use_id": content.id,
                    "input": tool_input,
                    "output": None,
                }

                if tool_name in tool_handlers:
                    if verbose:
                        print(f"\n🔧 Tool: {tool_name}")
                        if tool_name == "python_expression":
                            print(f"📥 Input:\n```python\n{tool_input['expression']}\n```")
                        else:
                            print(f"📥 Input: {tool_input}")

                    handler = tool_handlers[tool_name]

                    # Call handler with appropriate arguments
                    if tool_name == "python_expression":
                        result = handler(tool_input["expression"])
                    elif tool_name == "submit_predictions":
                        result = handler(predictions_file=tool_input.get("predictions_file", ""))
                        submitted_answer = result
                    else:
                        result = handler(**tool_input) if isinstance(tool_input, dict) else handler(tool_input)

                    if verbose:
                        print(f"📤 Output: {result}")

                    tool_call_log["output"] = result
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content.id,
                        "content": json.dumps(result),
                    })

                step_log["tool_calls"].append(tool_call_log)

        log_data["steps"].append(step_log)

        if has_tool_use:
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            if submitted_answer is not None:
                if verbose:
                    print(f"\n✅ Agent submitted answer!")
                log_data["final_result"] = submitted_answer
                log_data["completion_status"] = "submitted"
                if log_file:
                    _write_markdown_log(log_file, log_data)
                return submitted_answer
        else:
            if verbose:
                print("\n⚠️  No tool use in response, ending loop.")
            log_data["completion_status"] = "no_tool_use"
            break

    if verbose:
        print(f"\n⏱️  Reached maximum steps ({max_steps}) without submitting answer.")

    log_data["completion_status"] = "max_steps_reached"
    if log_file:
        _write_markdown_log(log_file, log_data)

    return None


def _write_markdown_log(log_file: Path, log_data: dict) -> None:
    """Write log data to a Markdown file."""
    md_file = log_file.with_suffix('.md')

    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(f"# Agent Run Log\n\n")
        f.write(f"**Timestamp:** {log_data['timestamp']}\n\n")
        f.write(f"**Model:** {log_data['model']}\n\n")
        f.write(f"**Max Steps:** {log_data['max_steps']}\n\n")
        f.write(f"**Completion Status:** {log_data['completion_status']}\n\n")
        f.write(f"## Initial Prompt\n\n```\n{log_data['initial_prompt']}\n```\n\n")
        f.write(f"## Execution Steps\n\n")

        for step in log_data['steps']:
            f.write(f"### Step {step['step_number']}\n\n")
            f.write(f"**API Response:**\n")
            f.write(f"- ID: `{step['api_response']['id']}`\n")
            f.write(f"- Stop Reason: `{step['api_response']['stop_reason']}`\n")
            f.write(f"- Input Tokens: {step['api_response']['usage']['input_tokens']}\n")
            f.write(f"- Output Tokens: {step['api_response']['usage']['output_tokens']}\n\n")

            for i, tool_call in enumerate(step['tool_calls']):
                if tool_call.get('type') == 'text':
                    f.write(f"**Assistant Response:**\n\n{tool_call['content']}\n\n")
                else:
                    f.write(f"#### Tool Call {i + 1}: `{tool_call['tool_name']}`\n\n")
                    f.write(f"**Input:**\n\n")

                    if tool_call['tool_name'] == 'python_expression':
                        f.write(f"```python\n{tool_call['input'].get('expression', '')}\n```\n\n")
                    else:
                        f.write(f"```json\n{json.dumps(tool_call['input'], indent=2)}\n```\n\n")

                    f.write(f"**Output:**\n\n")
                    if tool_call['output']:
                        f.write(f"```json\n{json.dumps(tool_call['output'], indent=2, default=str)}\n```\n\n")
                    else:
                        f.write(f"```\n(no output)\n```\n\n")

            f.write(f"---\n\n")

        f.write(f"## Final Result\n\n")
        if log_data['final_result']:
            f.write(f"```json\n{json.dumps(log_data['final_result'], indent=2, default=str)}\n```\n\n")
        else:
            f.write(f"*No result submitted*\n\n")

        if 'grading' in log_data:
            f.write(f"## Grading\n\n")
            f.write(f"**Success:** {log_data['grading']['success']}\n\n")
            if log_data['grading']['result']:
                f.write(f"**Result:**\n\n")
                f.write(f"```json\n{json.dumps(log_data['grading']['result'], indent=2, default=str)}\n```\n\n")


async def run_single_test(
    run_id: int,
    num_runs: int,
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    grading_func: Callable[[Any], tuple[bool, dict]],
    verbose: bool = False,
) -> tuple[int, bool, Any]:
    """Run a single test iteration."""
    reset_task_state()

    if verbose:
        print(f"\n\n{'=' * 20} RUN {run_id}/{num_runs} {'=' * 20}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"run_{run_id:03d}_{timestamp}.md"

    result = await run_agent_loop(
        prompt=prompt,
        tools=tools,
        tool_handlers=tool_handlers,
        max_steps=20,
        verbose=verbose,
        log_file=log_file,
    )

    success, metrics = grading_func(result)

    # Update log file with grading result
    md_file = log_file.with_suffix('.md')
    if md_file.exists():
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()

        if '## Grading' not in content:
            with open(md_file, 'a', encoding='utf-8') as f:
                f.write(f"## Grading\n\n")
                f.write(f"**Success:** {success}\n\n")

                if metrics:
                    f.write(f"**Metrics:**\n\n")
                    f.write(f"- True Positives: {metrics.get('true_positives', 0)}\n")
                    f.write(f"- False Positives: {metrics.get('false_positives', 0)}\n")
                    f.write(f"- False Negatives: {metrics.get('false_negatives', 0)}\n")
                    f.write(f"- Precision: {metrics.get('precision', 0):.4f}\n")
                    f.write(f"- Recall: {metrics.get('recall', 0):.4f}\n")
                    f.write(f"- F1 Score: {metrics.get('f1', 0):.4f}\n\n")

                if result:
                    f.write(f"**Result:**\n\n")
                    f.write(f"```json\n{json.dumps(result, indent=2, default=str)}\n```\n\n")

    print(f"\n{'='*60}")
    print(f"{'✅' if success else '❌'} Run {run_id}: {'SUCCESS' if success else 'FAILURE'}")
    print(f"{'='*60}\n")

    return run_id, success, result


async def main(concurrent: bool = True):
    """Run the test multiple times and track success rate."""
    num_runs = 10

    print(f"\n{'='*60}")
    print(f"🚀 Running {num_runs} test iterations {'concurrently' if concurrent else 'sequentially'}")
    print(f"{'='*60}\n")

    tasks = [
        run_single_test(
            run_id=i + 1,
            num_runs=num_runs,
            prompt=PROMPT,
            tools=TOOLS,
            tool_handlers=TOOL_HANDLERS,
            grading_func=grading_func,
            verbose=True,
        )
        for i in range(num_runs)
    ]

    if concurrent:
        results = []
        for coro in asyncio.as_completed(tasks):
            results.append(await coro)
    else:
        results = [await task for task in tasks]

    successes = sum(success for _, success, _ in results)
    pass_rate = (successes / num_runs) * 100

    print(f"\n{'='*60}")
    print("📊 Test Results Summary")
    print(f"{'='*60}")
    print(f"✅ Passed:    {successes}/{num_runs}")
    print(f"❌ Failed:    {num_runs - successes}/{num_runs}")
    print(f"📈 Pass Rate: {pass_rate:.1f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main(concurrent=False))