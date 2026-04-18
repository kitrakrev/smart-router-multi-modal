"""Execute tools and return results."""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.tools.medical_tools import MEDICAL_TOOLS

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Execute medical tools and return structured results."""

    def __init__(self) -> None:
        self._tools = dict(MEDICAL_TOOLS)

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool by name with the given arguments.

        Returns a result dict or an error dict.
        """
        tool_info = self._tools.get(tool_name)
        if tool_info is None:
            return {"error": f"Unknown tool: {tool_name}", "available": list(self._tools.keys())}

        try:
            fn = tool_info["function"]
            result = fn(**arguments)
            return {"status": "success", "result": result}
        except TypeError as e:
            return {"status": "error", "error": f"Invalid arguments: {e}", "expected": tool_info["parameters"]}
        except Exception as e:
            logger.error("Tool %s execution failed: %s", tool_name, e)
            return {"status": "error", "error": str(e)}

    def list_tools(self) -> list[dict[str, Any]]:
        """Return list of available tools with descriptions."""
        return [
            {
                "name": name,
                "description": info["description"],
                "parameters": info["parameters"],
            }
            for name, info in self._tools.items()
        ]

    def match_tools(self, query: str) -> list[str]:
        """Simple keyword-based tool matching for a query.

        Returns list of tool names that might be relevant to the query.
        In production, this would use embedding similarity against tool
        exemplars from config/tools.yaml.
        """
        query_lower = query.lower()
        matched = []

        # Keyword-based matching
        tool_keywords = {
            "drug_interaction_check": [
                "drug", "interaction", "contraindication", "medication", "combine",
                "safe with", "together", "mixing",
            ],
            "clinical_guideline": [
                "guideline", "recommendation", "protocol", "standard of care",
                "management", "treatment plan", "acc", "aha",
            ],
            "lab_reference": [
                "lab", "normal range", "reference", "level", "troponin",
                "creatinine", "tsh", "hemoglobin", "potassium", "glucose",
            ],
            "dosage_calculator": [
                "dose", "dosage", "how much", "mg/kg", "calculate",
                "weight-based", "pediatric dose",
            ],
            "icd_code_lookup": [
                "icd", "code", "diagnosis code", "billing code",
                "icd-10", "classification",
            ],
        }

        for tool_name, keywords in tool_keywords.items():
            for kw in keywords:
                if kw in query_lower:
                    matched.append(tool_name)
                    break

        return matched


# Singleton
tool_executor = ToolExecutor()
