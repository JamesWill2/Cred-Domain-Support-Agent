"""
Standalone FastMCP Client for Cred Lending Operations.

Establishes a client connection to the FastMCP server, invokes
check_loan_application_status across multiple loan records, and verifies
standardized Model Context Protocol (MCP) responses.
"""

from typing import Dict, List, Any
import asyncio
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastmcp import Client
from mcp_service.server import mcp_server


async def run_mcp_client(test_record_ids: List[str] = None) -> List[Dict[str, Any]]:
    """
    Execute MCP client-server round trip for multiple record IDs.

    Args:
        test_record_ids: List of record IDs to query (defaults to ['LOAN-1001', 'LOAN-1002', 'LOAN-1003']).

    Returns:
        List of standardized tool call results.
    """
    if test_record_ids is None:
        test_record_ids = ["LOAN-1001", "LOAN-1002", "LOAN-1003"]

    print("=================================================================")
    print(" FASTMCP CLIENT ROUND-TRIP DEMONSTRATION")
    print(" Connecting to Cred-Lending-Operations-MCP Server...")
    print("=================================================================")

    results = []

    async with Client(mcp_server) as client:
        # 1. Discover available tools
        tools = await client.list_tools()
        tool_names = [t.name for t in tools]
        print(f"Connected successfully. Discovered MCP Tools: {tool_names}\n")

        # 2. Invoke check_loan_application_status for multiple records
        for rid in test_record_ids:
            print(f"--> [MCP Call] Invoking 'check_loan_application_status' with record_id='{rid}'")
            call_result = await client.call_tool(
                name="check_loan_application_status",
                arguments={"record_id": rid},
            )

            # Standardized MCP result parsing
            content_text = call_result.content[0].text if call_result.content else "{}"
            structured_data = getattr(call_result, "structured_content", None) or getattr(call_result, "data", {})

            print(f"<-- [MCP Standard Response Received]:")
            print(f"    Raw Text Content:       {content_text}")
            print(f"    Structured Content:     {structured_data}")
            print(f"    Is Error:               {call_result.is_error}\n")

            results.append({
                "record_id": rid,
                "is_error": call_result.is_error,
                "structured_data": structured_data,
                "text_content": content_text,
            })

    print("=================================================================")
    print(f"Successfully completed {len(results)} MCP tool calls via protocol round-trip.")
    print("=================================================================\n")
    return results


def main():
    asyncio.run(run_mcp_client())


if __name__ == "__main__":
    main()

