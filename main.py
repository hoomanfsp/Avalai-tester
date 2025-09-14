#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import argparse
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
from typing import Tuple

# Optional .env loader
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None  # allow running without python-dotenv

# OpenAI-compatible SDK
from openai import OpenAI
from openai import APIConnectionError, APIStatusError, RateLimitError, APITimeoutError

LOG_FILE = "avalai.log"
CONV_FILE = "conv.txt"


# --------------------------- Logging ---------------------------------
def setup_logger() -> logging.Logger:
    logger = logging.getLogger("avalai")
    logger.setLevel(logging.INFO)

    handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Warn+ to stderr for quick visibility
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    stderr_handler.setLevel(logging.WARNING)
    logger.addHandler(stderr_handler)
    return logger


logger = setup_logger()


# --------------------------- Utils -----------------------------------
def isotime_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "t", "yes", "y", "on"}


def get_prompt_from_sources(cli_prompt: str | None) -> Tuple[str, str]:
    """
    Priority: CLI > stdin > .env (AVALAI_PROMPT)
    Returns: (prompt, source)
    """
    # 1) CLI
    if cli_prompt is not None and cli_prompt.strip():
        return cli_prompt.strip(), "cli"

    # 2) stdin if piped
    if not sys.stdin.isatty():
        data = sys.stdin.read().strip()
        if data:
            return data, "stdin"

    # 3) .env
    env_prompt = os.getenv("AVALAI_PROMPT", "").strip()
    if env_prompt:
        return env_prompt, ".env"

    return "", "none"


def get_client(timeout: float | None = None) -> OpenAI:
    base_url = os.getenv("AVALAI_BASE_URL")
    api_key = os.getenv("AVALAI_API_KEY")

    if not base_url or not api_key:
        msg = "Missing AVALAI_BASE_URL or AVALAI_API_KEY in environment (.env)."
        logger.error(msg)
        raise RuntimeError(msg)

    # openai>=1.x supports timeout on client
    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)


def build_messages(prompt: str, system: str | None, allow_system: bool) -> list[dict]:
    """Build chat messages honoring allow_system. If disallowed, inline system text."""
    if allow_system and system:
        return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    if system:
        inlined = (
            "### Instruction\n"
            f"{system}\n\n"
            "### Task\n"
            f"{prompt}"
        )
        return [{"role": "user", "content": inlined}]
    return [{"role": "user", "content": prompt}]


def is_dev_instruction_error(err: Exception) -> bool:
    """Detect provider error for disabled developer/system instructions."""
    msg = str(err)
    return ("Developer instruction is not enabled" in msg) or ("developer instruction" in msg.lower())


def save_conversation(prompt: str, response: str, filename: str = CONV_FILE) -> None:
    """Append a prompt+response block to conv.txt with timestamp."""
    ts = isotime_now()
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"\n====== {ts} ======\n")
        f.write("Prompt:\n")
        f.write(prompt.strip() + "\n\n")
        f.write("Response:\n")
        f.write(response.strip() + "\n")


# ----------------------- Core request logic --------------------------
def send_prompt(
    client: OpenAI,
    prompt: str,
    model: str,
    system: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    max_retries: int = 3,
    retry_backoff: float = 1.5,
    prompt_source: str = "unknown",
    allow_system: bool = True,
) -> str:
    """
    Sends a prompt via Chat Completions (OpenAI-compatible).
    Retries on transient errors, and auto-falls back if system messages are not allowed.
    """
    attempt = 0
    t0 = time.perf_counter()

    logger.info(json.dumps({
        "event": "request",
        "timestamp": isotime_now(),
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "prompt_source": prompt_source,
        "has_system": bool(system),
        "allow_system": bool(allow_system),
        "prompt_preview": prompt if len(prompt) <= 500 else (prompt[:500] + " …[truncated]"),
    }, ensure_ascii=False))

    last_err = None
    used_allow_system = allow_system
    fallback_used = False

    while attempt < max_retries:
        attempt += 1
        try:
            start = time.perf_counter()
            messages = build_messages(prompt, system, used_allow_system)
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            latency = time.perf_counter() - start

            # Extract text
            text = ""
            if resp and resp.choices:
                choice = resp.choices[0]
                if hasattr(choice, "message") and getattr(choice.message, "content", None) is not None:
                    text = choice.message.content or ""
                elif hasattr(choice, "text"):
                    text = choice.text or ""

            logger.info(json.dumps({
                "event": "response",
                "timestamp": isotime_now(),
                "attempt": attempt,
                "latency_sec": round(latency, 3),
                "usage": getattr(resp, "usage", None) and resp.usage.model_dump() or None,
                "id": getattr(resp, "id", None),
                "fallback_used": fallback_used,
                "allow_system_effective": used_allow_system,
            }, ensure_ascii=False))

            total_latency = time.perf_counter() - t0
            logger.info(json.dumps({
                "event": "summary",
                "timestamp": isotime_now(),
                "status": "success",
                "total_latency_sec": round(total_latency, 3),
            }, ensure_ascii=False))

            return text

        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            last_err = e
            wait = (retry_backoff ** (attempt - 1))
            logger.warning(json.dumps({
                "event": "retry",
                "timestamp": isotime_now(),
                "attempt": attempt,
                "error": str(e),
                "next_backoff_sec": round(wait, 2),
                "fallback_used": fallback_used,
            }, ensure_ascii=False))
            time.sleep(wait)

        except APIStatusError as e:
            # Detect "Developer instruction not enabled" and retry without system role
            last_err = e
            if is_dev_instruction_error(e) and used_allow_system:
                logger.warning(json.dumps({
                    "event": "fallback_disable_system",
                    "timestamp": isotime_now(),
                    "attempt": attempt,
                    "reason": "Developer instruction not enabled; retrying without system role by inlining.",
                }, ensure_ascii=False))
                used_allow_system = False
                fallback_used = True
                attempt -= 1  # immediate retry without consuming an attempt
                continue

            wait = (retry_backoff ** (attempt - 1))
            logger.warning(json.dumps({
                "event": "retry",
                "timestamp": isotime_now(),
                "attempt": attempt,
                "error": str(e),
                "next_backoff_sec": round(wait, 2),
                "fallback_used": fallback_used,
            }, ensure_ascii=False))
            time.sleep(wait)

        except Exception as e:
            last_err = e
            logger.error(json.dumps({
                "event": "fatal_error",
                "timestamp": isotime_now(),
                "attempt": attempt,
                "error": repr(e),
                "fallback_used": fallback_used,
            }, ensure_ascii=False))
            break

    total_latency = time.perf_counter() - t0
    logger.error(json.dumps({
        "event": "summary",
        "timestamp": isotime_now(),
        "status": "failed",
        "total_latency_sec": round(total_latency, 3),
        "error": str(last_err) if last_err else "unknown_error",
    }, ensure_ascii=False))
    raise SystemExit(f"Request failed after {max_retries} attempts: {last_err}")


# ---------------------------- CLI ------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a prompt to an OpenAI-compatible endpoint (e.g., AvalAI) and log everything to avalai.log; also append conv history to conv.txt."
    )
    parser.add_argument(
        "-p", "--prompt",
        help="Prompt text. If omitted, tries stdin, then AVALAI_PROMPT from .env.",
    )
    parser.add_argument(
        "-m", "--model",
        default=os.getenv("AVALAI_MODEL", "gpt-4o-mini"),
        help="Model name (default from AVALAI_MODEL env or 'gpt-4o-mini').",
    )
    parser.add_argument(
        "--system",
        default=os.getenv("AVALAI_SYSTEM", None),
        help="Optional system prompt (or set AVALAI_SYSTEM in .env).",
    )
    parser.add_argument(
        "--allow-system",
        type=str2bool,
        default=str2bool(os.getenv("AVALAI_ALLOW_SYSTEM", "true")),
        help="Allow sending system/developer messages (default true, or set AVALAI_ALLOW_SYSTEM).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("AVALAI_TEMPERATURE", "0.2")),
        help="Sampling temperature (default: 0.2 or AVALAI_TEMPERATURE).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.getenv("AVALAI_MAX_TOKENS", "1024")),
        help="Max tokens for the response (default: 1024 or AVALAI_MAX_TOKENS).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("AVALAI_TIMEOUT", "60")),
        help="Client timeout in seconds (default: 60 or AVALAI_TIMEOUT).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=int(os.getenv("AVALAI_RETRIES", "3")),
        help="Max retries on transient errors (default: 3 or AVALAI_RETRIES).",
    )
    return parser.parse_args()


def main() -> None:
    # Load .env early if available
    if load_dotenv:
        load_dotenv()

    args = parse_args()

    # Resolve prompt & source
    prompt, prompt_source = get_prompt_from_sources(args.prompt)
    if not prompt:
        print("No prompt provided. Use -p/--prompt, pipe via stdin, or set AVALAI_PROMPT in .env.", file=sys.stderr)
        sys.exit(1)

    # Create client
    client = get_client(timeout=args.timeout)

    # Send request
    text = send_prompt(
        client=client,
        prompt=prompt,
        model=args.model,
        system=args.system,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.retries,
        prompt_source=prompt_source,
        allow_system=args.allow_system,
    )

    # Output to stdout
    print(text)

    # Save to conversation file
    save_conversation(prompt, text)


if __name__ == "__main__":
    main()
