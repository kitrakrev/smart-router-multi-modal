"""Tool execution framework.

Provides simulated security and utility tools with realistic responses.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    output: dict
    execution_time_ms: float


# ---------------------------------------------------------------------------
# Tool definitions (for OpenAI-compatible tool_choice)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "block_ip",
            "description": "Block an IP address at the firewall level",
            "parameters": {
                "type": "object",
                "properties": {
                    "ip": {"type": "string", "description": "IP address to block"},
                    "duration_hours": {"type": "integer", "description": "Block duration in hours", "default": 24},
                },
                "required": ["ip"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scan_ports",
            "description": "Scan open ports on a target host",
            "parameters": {
                "type": "object",
                "properties": {
                    "host": {"type": "string", "description": "Target hostname or IP"},
                    "port_range": {"type": "string", "description": "Port range, e.g. '1-1024'", "default": "1-1024"},
                },
                "required": ["host"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_reputation",
            "description": "Check the reputation score of an IP or domain",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "IP address or domain to check"},
                },
                "required": ["target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "quarantine_host",
            "description": "Isolate a host from the network for investigation",
            "parameters": {
                "type": "object",
                "properties": {
                    "host": {"type": "string", "description": "Host to quarantine"},
                    "reason": {"type": "string", "description": "Reason for quarantine"},
                },
                "required": ["host"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_report",
            "description": "Generate a security incident report",
            "parameters": {
                "type": "object",
                "properties": {
                    "incident_type": {"type": "string", "description": "Type of incident"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                    "details": {"type": "string", "description": "Incident details"},
                },
                "required": ["incident_type", "severity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_firewall_rule",
            "description": "Add or modify a firewall rule",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["allow", "deny", "drop"]},
                    "source": {"type": "string", "description": "Source IP/CIDR"},
                    "destination": {"type": "string", "description": "Destination IP/CIDR"},
                    "port": {"type": "integer", "description": "Port number"},
                },
                "required": ["action", "source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_vulnerability_scan",
            "description": "Run a vulnerability scan on a target",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "Target to scan"},
                    "scan_type": {"type": "string", "enum": ["quick", "full", "compliance"], "default": "quick"},
                },
                "required": ["target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_cve",
            "description": "Look up a CVE by ID and return details",
            "parameters": {
                "type": "object",
                "properties": {
                    "cve_id": {"type": "string", "description": "CVE identifier, e.g. CVE-2024-1234"},
                },
                "required": ["cve_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "num_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression to evaluate"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "code_executor",
            "description": "Execute a code snippet in a sandboxed environment",
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {"type": "string", "enum": ["python", "javascript", "bash"]},
                    "code": {"type": "string", "description": "Code to execute"},
                },
                "required": ["language", "code"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Simulated tool implementations
# ---------------------------------------------------------------------------

def _sim_block_ip(args: dict) -> dict:
    ip = args.get("ip", "0.0.0.0")
    hours = args.get("duration_hours", 24)
    return {"status": "blocked", "ip": ip, "duration_hours": hours,
            "rule_id": f"FW-{random.randint(10000,99999)}",
            "message": f"IP {ip} blocked for {hours}h"}


def _sim_scan_ports(args: dict) -> dict:
    host = args.get("host", "unknown")
    open_ports = sorted(random.sample([22, 80, 443, 3306, 5432, 8080, 8443, 9090, 27017], k=random.randint(2, 5)))
    return {"host": host, "open_ports": open_ports,
            "scan_duration_ms": random.randint(200, 2000),
            "total_scanned": 1024}


def _sim_check_reputation(args: dict) -> dict:
    target = args.get("target", "unknown")
    score = round(random.uniform(0, 100), 1)
    category = "clean" if score > 70 else "suspicious" if score > 30 else "malicious"
    return {"target": target, "reputation_score": score, "category": category,
            "reports": random.randint(0, 50), "first_seen": "2023-06-15",
            "tags": random.sample(["scanner", "botnet", "tor_exit", "vpn", "proxy", "clean"], k=2)}


def _sim_quarantine_host(args: dict) -> dict:
    return {"host": args.get("host"), "status": "quarantined",
            "vlan": "QUARANTINE_VLAN_999",
            "reason": args.get("reason", "investigation"),
            "ticket_id": f"INC-{random.randint(1000,9999)}"}


def _sim_generate_report(args: dict) -> dict:
    return {"report_id": f"RPT-{random.randint(10000,99999)}",
            "incident_type": args.get("incident_type"),
            "severity": args.get("severity", "medium"),
            "status": "generated",
            "pdf_url": f"/reports/RPT-{random.randint(10000,99999)}.pdf",
            "summary": f"Security incident report for {args.get('incident_type', 'unknown')} generated successfully."}


def _sim_update_firewall(args: dict) -> dict:
    return {"rule_id": f"FW-{random.randint(10000,99999)}",
            "action": args.get("action"), "source": args.get("source"),
            "status": "applied", "effective_at": "2025-01-01T00:00:00Z"}


def _sim_vuln_scan(args: dict) -> dict:
    vulns = [
        {"id": "CVE-2024-3094", "severity": "critical", "package": "xz-utils"},
        {"id": "CVE-2024-21762", "severity": "high", "package": "FortiOS"},
        {"id": "CVE-2023-44487", "severity": "high", "package": "HTTP/2"},
    ]
    found = random.sample(vulns, k=random.randint(0, 3))
    return {"target": args.get("target"), "vulnerabilities_found": len(found),
            "results": found, "scan_type": args.get("scan_type", "quick")}


def _sim_lookup_cve(args: dict) -> dict:
    cve_id = args.get("cve_id", "CVE-0000-0000")
    return {"cve_id": cve_id, "cvss_score": round(random.uniform(3, 10), 1),
            "severity": random.choice(["medium", "high", "critical"]),
            "description": f"Remote code execution vulnerability in {random.choice(['OpenSSL', 'Apache', 'nginx', 'kernel'])}",
            "published": "2024-03-15", "references": [f"https://nvd.nist.gov/vuln/detail/{cve_id}"]}


def _sim_web_search(args: dict) -> dict:
    query = args.get("query", "")
    return {"query": query, "results": [
        {"title": f"Result about {query}", "url": f"https://example.com/{i}", "snippet": f"Information about {query}..."}
        for i in range(args.get("num_results", 3))
    ]}


def _sim_calculator(args: dict) -> dict:
    expr = args.get("expression", "0")
    try:
        result = eval(expr, {"__builtins__": {}}, {})
        return {"expression": expr, "result": result}
    except Exception as e:
        return {"expression": expr, "error": str(e)}


def _sim_code_executor(args: dict) -> dict:
    return {"language": args.get("language", "python"),
            "status": "executed",
            "stdout": f"[simulated output for {args.get('language', 'python')} code]",
            "stderr": "", "exit_code": 0, "execution_time_ms": random.randint(10, 500)}


_EXECUTORS = {
    "block_ip": _sim_block_ip,
    "scan_ports": _sim_scan_ports,
    "check_reputation": _sim_check_reputation,
    "quarantine_host": _sim_quarantine_host,
    "generate_report": _sim_generate_report,
    "update_firewall_rule": _sim_update_firewall,
    "run_vulnerability_scan": _sim_vuln_scan,
    "lookup_cve": _sim_lookup_cve,
    "web_search": _sim_web_search,
    "calculator": _sim_calculator,
    "code_executor": _sim_code_executor,
}


class ToolExecutor:
    """Execute a tool by name with given arguments."""

    @staticmethod
    def available_tools() -> list[str]:
        return list(_EXECUTORS.keys())

    @staticmethod
    def get_definitions() -> list[dict]:
        return TOOL_DEFINITIONS

    @staticmethod
    def execute(tool_name: str, arguments: dict) -> ToolResult:
        t0 = time.perf_counter()
        executor = _EXECUTORS.get(tool_name)
        if not executor:
            return ToolResult(
                tool_name=tool_name, success=False,
                output={"error": f"Unknown tool: {tool_name}"},
                execution_time_ms=0,
            )
        try:
            output = executor(arguments)
            elapsed = round((time.perf_counter() - t0) * 1000, 2)
            return ToolResult(tool_name=tool_name, success=True, output=output, execution_time_ms=elapsed)
        except Exception as e:
            elapsed = round((time.perf_counter() - t0) * 1000, 2)
            return ToolResult(tool_name=tool_name, success=False, output={"error": str(e)}, execution_time_ms=elapsed)
