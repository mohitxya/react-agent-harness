"""The Part 1 coding agent: fix a software issue and submit a git patch."""

from __future__ import annotations

import json
from typing import Any

from assignment.agent.base import (
    DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
    DEFAULT_COMPACTION_MAX_TOKENS,
    Agent,
    format_tool_output,
)

from assignment.agent.tools import EXECUTE_TOOL, SEND_MESSAGE_TOOL
from assignment.env import Environment

class CodeAgent(Agent):
    """An agent that fixes a software issue and submits a git patch."""

    def __init__(
        self,
        task: str,
        environment: Environment,
        model: str | None = None,
        logs_save_path: str | None = None,
        step_limit: int = 100,
        skills_path: str | None = None,
        auto_stop_environment: bool = True,
        compact_threshold_tokens: int | None = None,
        compaction_keep_recent_steps: int = DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
        compaction_max_tokens: int = DEFAULT_COMPACTION_MAX_TOKENS,
    ):
        super().__init__(
            environment=environment,
            model=model,
            logs_save_path=logs_save_path,
            step_limit=step_limit,
            skills_path=skills_path,
            auto_stop_environment=auto_stop_environment,
            compact_threshold_tokens=compact_threshold_tokens,
            compaction_keep_recent_steps=compaction_keep_recent_steps,
            compaction_max_tokens=compaction_max_tokens,
        )
        self.task = task
        self.submitted_patch = ""

        # TODO(Part 1.3): Make the `execute` and `send_message` tools available
        # to the agent.
        self.tools.extend([EXECUTE_TOOL, SEND_MESSAGE_TOOL])

        # TODO(1.1.b): Construct the system prompt and task_prompt. These
        # should be usable by the `Agent.build_prompt` method.
        self.task_prompt = self.task
        sys_info = json.dumps(
        {
            "machine": self.env.machine,
            "release": self.env.release,
            "system": self.env.system,
            "version": self.env.version,
        },
        indent=2,
        )
        self.system_prompt = (
            "You are a helpful software engineer assistant. You have access to bash "
            "terminal tools to investigate, reproduce, and fix software issues in the repository. "
            "Work inside the testbed, verify your fix with tests, and ensure you submit your work.\n\n"
            f"<system_information>\n{sys_info}\n</system_information>"
        )


        # TODO(1.4): If any skills are available to the agent, make their
        # descriptions/metadata available to the agent in the prompt.

        if self.skills: 
            catalog = "\n".join(skill["metadata"] for skill in self.skills.values())
            self.system_prompt += ( "\n\nReusable skills are available. Call `invoke_skill` with a skill's name "
        f"to load its instructions, and follow them in place of your default approach.\n\n"
        f"<skills>\n{catalog}\n</skills>\n")

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute ``execute`` and ``send_message`` calls in the code sandbox."""

        # TODO(Part 1.3): Parse each call, execute recognized tools, and return
        # one message per call (there may be multiple tool calls in one agent
        # response!). Malformed JSON and unknown tools must become recoverable
        # observations relayed to the agent instead of exceptions.
        observations: list[dict[str, str]] = []
        for call in tool_calls:
            call_id = call.get("id", "")
            func = call.get("function", {})
            name = func.get("name", "")
            raw_args = func.get("arguments", "{}")
            # 1. Safely parse JSON arguments
            try:
                if isinstance(raw_args, str):
                    args = json.loads(raw_args)
                elif isinstance(raw_args, dict):
                    args = raw_args
                else:
                    args = {}
                if not isinstance(args, dict):
                    raise ValueError("Tool arguments must be a JSON object.")
            except Exception as exc:
                observations.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": f"<error>Malformed arguments: {exc}</error>",
                })
                continue
            # 2. Dispatch on tool name
            if name == "execute":
                command = args.get("command")
                if command is None:
                    content = "<error>Missing required 'command' argument</error>"
                else:
                    result = self.env.execute(
                        command=command,
                        cwd=args.get("cwd"),
                        env=args.get("env"),
                        timeout=args.get("timeout"),
                        shell=args.get("shell"),
                    )
                    content = format_tool_output(result)
            elif name == "send_message":
                summary = args.get("summary", "")
                self.submitted_patch = summary
                self.finished = True
                content = f"<message_sent>{summary}</message_sent>"
            elif name == "invoke_skill":
                skill_name = args.get("name", "")
                if skill_name in self.skills:
                    content = self.skills[skill_name]["content"]
                else:
                    content = f"<error>Skill '{skill_name}' not found</error>"
            else:
                content = f"<error>Unknown tool: {name}</error>"
            # 3. Link the observation to this call ID
            observations.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": content,
            })
        return observations