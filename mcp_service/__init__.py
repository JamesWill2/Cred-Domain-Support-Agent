"""
FastMCP Service Package for Cred Domain Support Agent.
Exposes lending operations tools via the Model Context Protocol (MCP).
"""
from mcp_service.server import mcp_server

__all__ = ["mcp_server"]

