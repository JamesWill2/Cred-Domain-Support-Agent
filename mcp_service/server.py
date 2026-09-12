"""
FastMCP Server for Cred Lending Operations.

Wraps check_loan_application_status as an MCP tool with full parameter and
return value docstrings, exposing it over standard FastMCP transports.
"""

from typing import Dict, Any
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastmcp import FastMCP
from agent.tools import check_loan_application_status as _check_status

# Initialize FastMCP Server
mcp_server = FastMCP(
    name="Cred-Lending-Operations-MCP",
)


@mcp_server.tool(name="check_loan_application_status")
def check_loan_application_status(record_id: str) -> Dict[str, Any]:
    """
    Look up a loan application record by ID (e.g. 'LOAN-1001') and evaluate its escalation score.

    Parameters:
        record_id: Unique loan application identifier (e.g., 'LOAN-1001', 'LOAN-1002').

    Returns:
        A structured dictionary containing:
        - record_id: Loan application ID.
        - found: Boolean whether application exists in Cred database.
        - category: Category of loan (Personal, Home, Auto, Education, Business).
        - status: Current workflow status (Submitted, Under Review, Approved, Rejected, Disbursed).
        - loan_amount_inr: Principal approved amount in Indian Rupees.
        - days_since_created: Application age in days (0-30).
        - flagged_for_fraud_review: Whether flagged by fraud risk systems.
        - escalation_score: Continuous multi-factor escalation score in [0.0, 1.0].
        - escalation_threshold: Operational escalation cut-off (0.65).
        - escalation_required: Boolean flag indicating if human intervention is required.
        - message: Natural language summary of the record and escalation recommendation.
    """
    return _check_status(record_id)


if __name__ == "__main__":
    mcp_server.run()

