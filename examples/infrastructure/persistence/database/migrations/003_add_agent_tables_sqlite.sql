-- Migration 003: Add agent orchestration tables (SQLite version)

-- Agent state persistence
CREATE TABLE IF NOT EXISTS agent_states (
    agent_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    agent_type TEXT NOT NULL,
    state TEXT NOT NULL, -- JSON as TEXT in SQLite
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Proposal tracking
CREATE TABLE IF NOT EXISTS agent_proposals (
    proposal_id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL REFERENCES agent_states(agent_id) ON DELETE CASCADE,
    proposal_type TEXT NOT NULL,
    content TEXT NOT NULL, -- JSON as TEXT in SQLite
    status TEXT DEFAULT 'PENDING',
    approved_by TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    executed_at DATETIME
);

-- Agent performance metrics
CREATE TABLE IF NOT EXISTS agent_metrics (
    metric_id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL REFERENCES agent_states(agent_id) ON DELETE CASCADE,
    metric_type TEXT NOT NULL,
    metric_value TEXT NOT NULL, -- JSON as TEXT in SQLite
    recorded_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_agent_states_project ON agent_states(project_id);
CREATE INDEX IF NOT EXISTS idx_agent_states_type ON agent_states(agent_type);
CREATE INDEX IF NOT EXISTS idx_agent_proposals_agent ON agent_proposals(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_proposals_status ON agent_proposals(status);
CREATE INDEX IF NOT EXISTS idx_agent_metrics_agent ON agent_metrics(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_metrics_type ON agent_metrics(metric_type);
