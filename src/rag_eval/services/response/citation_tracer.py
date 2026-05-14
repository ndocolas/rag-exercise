from __future__ import annotations

import re

_CITATION_TOKEN = re.compile(r"\[(\d+)\]")


class CitationTracer:
    """Markdown helpers for citation tokens.

    Scope intentionally tiny: the only consumer is the markdown view of
    `/ask`, which bolds `[N]` so citations stand out in prose.
    """

    @staticmethod
    def highlight_citations(answer: str) -> str:
        """Wrap `[N]` tokens in markdown bold so they pop in rendered text."""
        return _CITATION_TOKEN.sub(r"**[\1]**", answer)
