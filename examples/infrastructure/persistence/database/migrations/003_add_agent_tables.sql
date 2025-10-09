-- Migration 003: Add agent orchestration tables
-- Agent state persistence
CREATE TABLE agent_states (
    agent_id UUID PRIMARY KEY,
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    agent_type VARCHAR(100) NOT NULL,
    state JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Proposal tracking
CREATE TABLE agent_proposals (
    proposal_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES agent_states(agent_id) ON DELETE CASCADE,
    proposal_type VARCHAR(100) NOT NULL,
    content JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'PENDING',
    approved_by UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    executed_at TIMESTAMP WITH TIME ZONE
);

-- Agent performance metrics
CREATE TABLE agent_metrics (
    metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES agent_states(agent_id) ON DELETE CASCADE,
    metric_type VARCHAR(100) NOT NULL,
    metric_value JSONB NOT NULL,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_agent_states_project ON agent_states(project_id);
CREATE INDEX idx_agent_states_type ON agent_states(agent_type);
CREATE INDEX idx_agent_proposals_agent ON agent_proposals(agent_id);
CREATE INDEX idx_agent_proposals_status ON agent_proposals(status);
CREATE INDEX idx_agent_metrics_agent ON agent_metrics(agent_id);
CREATE INDEX idx_agent_metrics_type ON agent_metrics(metric_type);
