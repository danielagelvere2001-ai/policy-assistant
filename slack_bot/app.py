"""Slack bot that answers company policy questions and names the relevant policy.

Runs locally in Socket Mode (no public URL needed):

    python -m slack_bot.app

Ways to call it in Slack:
  * mention it in a channel:   @Policy Bot how many vacation days do I get?
  * the slash command:         /policy how many vacation days do I get?
  * a direct message to the bot

It uses the "LLM with vector index" approach by default (set BOT_APPROACH=rules
or BOT_APPROACH=llm in .env to switch).
"""
from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from assistant import build_all  # noqa: E402
from assistant.policies import find_policy, load_policies  # noqa: E402

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("policy-bot")

POLICIES = load_policies()
APPROACH_KEY = os.environ.get("BOT_APPROACH", "vector")
APPROACH = build_all(POLICIES)[APPROACH_KEY]
log.info("Using approach: %s", APPROACH.name)

app = App(token=os.environ["SLACK_BOT_TOKEN"])

HELP = ("Ask me about a company policy, for example: "
        "*@Policy Bot how many vacation days do I get?* or `/policy can I work from home?`")


def answer_blocks(question: str) -> tuple[str, list[dict]]:
    """Run the question through the assistant and format a Slack message."""
    a = APPROACH.ask(question)
    p = find_policy(POLICIES, a.policy_title)
    if p:
        policy_line = (f":page_facing_up: *Relevant policy:* {p.title}  ·  {p.category}  ·  "
                       f"owner: {p.department}\n> {p.text}")
    else:
        policy_line = ":mag: *Relevant policy:* none found in the policy database"
    footer = f"{APPROACH.name} · {a.time_ms / 1000:.1f}s · {a.total_tokens:,} tokens"
    blocks = [
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*Q:* {question}\n\n{a.answer}"}},
        {"type": "section", "text": {"type": "mrkdwn", "text": policy_line}},
        {"type": "context", "elements": [{"type": "mrkdwn", "text": footer}]},
    ]
    fallback = f"{a.answer} (Relevant policy: {a.policy_title or 'none'})"
    return fallback, blocks


def clean(text: str) -> str:
    return re.sub(r"<@[A-Z0-9]+>", "", text or "").strip()


def safe_answer(question: str) -> tuple[str, list[dict] | None]:
    if not question:
        return HELP, None
    try:
        return answer_blocks(question)
    except Exception as exc:  # keep the bot alive and tell the user
        log.exception("Failed to answer")
        return f":warning: Sorry, something went wrong while answering: {exc}", None


@app.event("app_mention")
def on_mention(event, say):
    text, blocks = safe_answer(clean(event.get("text", "")))
    say(text=text, blocks=blocks, thread_ts=event.get("thread_ts") or event["ts"])


@app.event("message")
def on_dm(event, say):
    # Only answer direct messages from people (ignore bot messages and edits).
    if event.get("channel_type") != "im" or event.get("bot_id") or event.get("subtype"):
        return
    text, blocks = safe_answer(clean(event.get("text", "")))
    say(text=text, blocks=blocks)


@app.command("/policy")
def on_command(ack, command, respond):
    ack()
    text, blocks = safe_answer(command.get("text", "").strip())
    respond(text=text, blocks=blocks, response_type="in_channel")


if __name__ == "__main__":
    log.info("Policy bot is running. Press Ctrl+C to stop.")
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
