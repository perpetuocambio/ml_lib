#!/bin/bash

# One-command MCP server installation
echo "🚀 Installing Claude Code MCP Servers..."

# Install Sequential Thinking MCP
echo "📊 Setting up Sequential Thinking..."
claude mcp add sequential-thinking -s user -- npx -y @modelcontextprotocol/server-sequential-thinking

# Install Filesystem MCP (customize directories as needed)
echo "📁 Setting upwdp Filesystem access..."
claude mcp add filesystem -s user -- npx -y @modelcontextprotocol/server-filesystem /src/perpetuocambio/rip

# Install Puppeteer MCP
echo "🌐 Setting up Puppeteer browser automation..."
claude mcp add puppeteer -s user -- npx -y @modelcontextprotocol/server-puppeteer

# Install Web Fetching MCP
echo "🔍 Setting up Web Fetching..."
claude mcp add fetch -s user -- npx -y @kazuph/mcp-fetch


# Install Web Server Memory
echo "🔍 Setting up server Memory..."
claude mcp add memory -s user -- npx -y @modelcontextprotocol/server-memory@latest

# Verify installation
echo "✅ Verifying installation..."
claude mcp list


echo "🎉 Basic MCP servers installed successfully!"
echo "📝 For API key-based servers (Brave Search, Firecrawl) and Browser Tools, see the README for individual setup instructions."