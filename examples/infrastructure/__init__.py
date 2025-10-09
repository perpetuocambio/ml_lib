"""
Infrastructure module for PyIntelCivil.

Provides technical infrastructure organized by architectural concerns:
- config: Centralized configuration management
- persistence: Database, storage and caching
- communication: HTTP, WebSocket, MCP protocols
- processing: Document extraction, web scraping, analytics
- providers: External service providers (LLM, search)
"""

# Export only the essential infrastructure namespace
# Individual imports should be done by consumers as needed
