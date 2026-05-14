#!/usr/bin/env python3
"""
Research agent powered by Gemini 1.5 Flash function calling and Tavily Search.

Install:
    pip install google-generativeai tavily-python

Environment:
    GEMINI_API_KEY=your_google_ai_studio_key
    TAVILY_API_KEY=your_tavily_key

Example:
    python research_agent.py "recent breakthroughs in solid-state batteries" -o report.md
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover - depends on local environment.
    genai = None

try:
    from tavily import TavilyClient
except ImportError:  # pragma: no cover - depends on local environment.
    TavilyClient = None


MODEL_NAME = "gemini-1.5-flash"
MIN_SEARCHES = 3
MAX_SEARCHES = 5
MAX_TOOL_TURNS = 8
RETRYABLE_ERROR_MARKERS = (
    "429",
    "rate",
    "quota",
    "resource_exhausted",
    "temporarily",
    "timeout",
    "deadline",
    "503",
    "500",
)


@dataclass
class SearchRecord:
    query: str
    answer: str | None = None
    results: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


def retry_call(label: str, func, *, attempts: int = 3, base_delay: float = 4.0) -> Any:
    """Run an API call with small jittered delays for free-tier/rate-limit friendliness."""
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001 - API clients raise several exception types.
            last_error = exc
            text = str(exc).lower()
            retryable = any(marker in text for marker in RETRYABLE_ERROR_MARKERS)
            if attempt == attempts or not retryable:
                raise

            delay = base_delay * attempt + random.uniform(0.25, 1.25)
            print(
                f"{label} failed on attempt {attempt}/{attempts}; retrying in {delay:.1f}s...",
                file=sys.stderr,
            )
            time.sleep(delay)

    raise RuntimeError(f"{label} failed") from last_error


class ResearchAgent:
    def __init__(
        self,
        gemini_api_key: str,
        tavily_api_key: str,
        *,
        max_searches: int = MAX_SEARCHES,
        min_searches: int = MIN_SEARCHES,
    ) -> None:
        if not (MIN_SEARCHES <= min_searches <= max_searches <= MAX_SEARCHES):
            raise ValueError("Search bounds must stay within 3-5 searches.")

        genai.configure(api_key=gemini_api_key)
        self.tavily = TavilyClient(api_key=tavily_api_key)
        self.max_searches = max_searches
        self.min_searches = min_searches
        self.searches: list[SearchRecord] = []

        self.model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            tools=[self._search_tool_declaration()],
        )

    @staticmethod
    def _search_tool_declaration() -> dict[str, Any]:
        return {
            "function_declarations": [
                {
                    "name": "web_search",
                    "description": (
                        "Search the web for authoritative, recent, or historical sources "
                        "needed to research the user's topic. Use focused, non-overlapping "
                        "queries. The tool returns concise Tavily results with URLs."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Focused search query.",
                            },
                            "search_depth": {
                                "type": "string",
                                "description": "Tavily search depth.",
                                "enum": ["basic", "advanced"],
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Number of results to return, from 3 to 8.",
                            },
                        },
                        "required": ["query"],
                    },
                }
            ]
        }

    def web_search(
        self,
        query: str,
        search_depth: str = "advanced",
        max_results: int = 5,
    ) -> dict[str, Any]:
        if len(self.searches) >= self.max_searches:
            return {
                "error": f"Search budget exhausted. Maximum is {self.max_searches}.",
                "searches_used": len(self.searches),
            }

        clean_query = " ".join(query.strip().split())
        if not clean_query:
            return {"error": "Empty query rejected."}

        depth = search_depth if search_depth in {"basic", "advanced"} else "advanced"
        result_limit = min(max(int(max_results or 5), 3), 8)

        try:
            response = retry_call(
                "Tavily search",
                lambda: self.tavily.search(
                    query=clean_query,
                    search_depth=depth,
                    max_results=result_limit,
                    include_answer=True,
                    include_raw_content=False,
                ),
                attempts=3,
                base_delay=3.0,
            )
        except Exception as exc:  # noqa: BLE001
            record = SearchRecord(query=clean_query, error=str(exc))
            self.searches.append(record)
            return {
                "query": clean_query,
                "error": str(exc),
                "searches_used": len(self.searches),
            }

        normalized_results = []
        for item in response.get("results", []):
            normalized_results.append(
                {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "content": item.get("content"),
                    "score": item.get("score"),
                    "published_date": item.get("published_date"),
                }
            )

        record = SearchRecord(
            query=clean_query,
            answer=response.get("answer"),
            results=normalized_results,
        )
        self.searches.append(record)

        return {
            "query": clean_query,
            "answer": response.get("answer"),
            "results": normalized_results,
            "searches_used": len(self.searches),
            "searches_remaining": self.max_searches - len(self.searches),
        }

    def run(self, topic: str) -> str:
        chat = self.model.start_chat()
        prompt = self._research_prompt(topic)

        response = retry_call(
            "Gemini research turn",
            lambda: chat.send_message(prompt),
            attempts=3,
            base_delay=5.0,
        )

        for _ in range(MAX_TOOL_TURNS):
            calls = self._extract_function_calls(response)
            if not calls:
                report = self._response_text(response)
                if self._is_usable_report(report):
                    return self._append_source_log(report)
                response = self._ask_for_final_report(chat)
                continue

            tool_parts = []
            for call in calls:
                if call["name"] != "web_search":
                    payload = {"error": f"Unknown function: {call['name']}"}
                else:
                    payload = self.web_search(**call["args"])

                tool_parts.append(self._function_response_part(call["name"], payload))

            time.sleep(1.5)
            response = retry_call(
                "Gemini tool-response turn",
                lambda: chat.send_message(tool_parts),
                attempts=3,
                base_delay=5.0,
            )

            if len(self.searches) >= self.max_searches:
                response = self._ask_for_final_report(chat)

        fallback_report = self._fallback_report(topic)
        return self._append_source_log(fallback_report)

    def _research_prompt(self, topic: str) -> str:
        today = datetime.now(timezone.utc).date().isoformat()
        return f"""
You are a careful research agent using web_search for live evidence.

Topic: {topic}
Date: {today}

Use Gemini native function calling to perform between {self.min_searches} and {self.max_searches}
focused web searches. Make the searches distinct: background, latest breakthroughs,
key organizations/people, and timeline/milestones are good angles.

After searching, synthesize a Markdown report with:

# {topic}

## Executive Summary
## Key Breakthroughs
## Timeline
## Current State
## Open Questions and Risks
## Sources

Requirements:
- Do not invent citations.
- Include source links in Markdown.
- Mention uncertainty where sources conflict or evidence is thin.
- Keep the report concise but useful.
- If a search fails, continue with remaining evidence and note the limitation.
""".strip()

    def _ask_for_final_report(self, chat) -> Any:
        if len(self.searches) < self.min_searches:
            n_needed = self.min_searches - len(self.searches)
            instruction = (
                f"You have only completed {len(self.searches)} searches. "
                f"Call web_search {n_needed} more time(s) before writing the report."
            )
        else:
            instruction = (
                "Now write the final Markdown report using only the gathered search evidence. "
                "Include breakthroughs, a timeline, and Markdown source links."
            )

        return retry_call(
            "Gemini final-report turn",
            lambda: chat.send_message(instruction),
            attempts=3,
            base_delay=5.0,
        )

    @staticmethod
    def _function_response_part(name: str, payload: dict[str, Any]) -> Any:
        response = {"result": payload}
        try:
            return genai.protos.Part(
                function_response=genai.protos.FunctionResponse(
                    name=name,
                    response=response,
                )
            )
        except AttributeError:
            return {
                "function_response": {
                    "name": name,
                    "response": response,
                }
            }

    @staticmethod
    def _extract_function_calls(response: Any) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        for part in getattr(response, "parts", []) or []:
            function_call = getattr(part, "function_call", None)
            if not function_call:
                continue
            name = getattr(function_call, "name", "")
            args = dict(getattr(function_call, "args", {}) or {})
            calls.append({"name": name, "args": args})
        return calls

    @staticmethod
    def _response_text(response: Any) -> str:
        text = getattr(response, "text", None)
        if text:
            return text.strip()

        chunks = []
        for part in getattr(response, "parts", []) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                chunks.append(part_text)
        return "\n".join(chunks).strip()

    @staticmethod
    def _is_usable_report(text: str) -> bool:
        required_sections = ("## Executive Summary", "## Key Breakthroughs", "## Timeline")
        return bool(text) and all(section in text for section in required_sections)

    def _fallback_report(self, topic: str) -> str:
        source_lines = []
        for search in self.searches:
            source_lines.append(f"- Query: {search.query}")
            if search.error:
                source_lines.append(f"  - Error: {search.error}")
                continue
            for item in search.results[:3]:
                title = item.get("title") or item.get("url") or "Source"
                url = item.get("url") or ""
                source_lines.append(f"  - [{title}]({url})")

        return f"""# {topic}

## Executive Summary
The agent could not get Gemini to produce a complete final report, but it did gather search evidence.

## Key Breakthroughs
Review the source log below and rerun the agent if needed.

## Timeline
Timeline synthesis was not completed because the model stopped before a valid final report.

## Current State
Searches completed: {len(self.searches)}

## Open Questions and Risks
The fallback report is incomplete and should not be treated as a finished research product.

## Sources
{chr(10).join(source_lines) if source_lines else "- No successful searches were completed."}
"""

    def _append_source_log(self, report: str) -> str:
        log = {
            "searches_used": len(self.searches),
            "queries": [
                {
                    "query": search.query,
                    "error": search.error,
                    "result_count": len(search.results),
                }
                for search in self.searches
            ],
        }
        return f"{report.rstrip()}\n\n<!-- Research agent search log:\n{json.dumps(log, indent=2)}\n-->\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Research a topic with Gemini 1.5 Flash and Tavily Search.",
    )
    parser.add_argument("topic", help="Research topic to investigate.")
    parser.add_argument(
        "-o",
        "--output",
        help="Optional Markdown output file. Prints to stdout if omitted.",
    )
    parser.add_argument(
        "--min-searches",
        type=int,
        default=MIN_SEARCHES,
        help="Minimum searches to perform. Must be 3-5.",
    )
    parser.add_argument(
        "--max-searches",
        type=int,
        default=MAX_SEARCHES,
        help="Maximum searches to perform. Must be 3-5.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if genai is None:
        print("Missing dependency: pip install google-generativeai", file=sys.stderr)
        return 2
    if TavilyClient is None:
        print("Missing dependency: pip install tavily-python", file=sys.stderr)
        return 2

    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")

    if not gemini_key:
        print("Missing GEMINI_API_KEY or GOOGLE_API_KEY.", file=sys.stderr)
        return 2
    if not tavily_key:
        print("Missing TAVILY_API_KEY.", file=sys.stderr)
        return 2

    try:
        agent = ResearchAgent(
            gemini_api_key=gemini_key,
            tavily_api_key=tavily_key,
            min_searches=args.min_searches,
            max_searches=args.max_searches,
        )
        report = agent.run(args.topic)
    except Exception as exc:  # noqa: BLE001
        print(f"Research agent failed: {exc}", file=sys.stderr)
        return 1

    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            file.write(report)
        print(f"Wrote report to {args.output}")
    else:
        print(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
