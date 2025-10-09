-- Migration 004: Add Dynamic Agent Configuration Tables
-- EPIC-009: Dynamic Agent Configuration System
-- Created: 2025-09-19
-- Description: Database schema for user-configurable dynamic agents

-- =====================================================
-- AGENT CONFIGURATIONS TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS agent_configurations (
    agent_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    name VARCHAR(255) NOT NULL,
    role_description TEXT NOT NULL,
    system_prompt TEXT NOT NULL,
    capabilities JSONB NOT NULL DEFAULT '[]',
    autonomy_level JSONB NOT NULL DEFAULT '{}',
    knowledge_context TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    last_modified TIMESTAMP DEFAULT NOW(),
    performance_metrics JSONB DEFAULT '{
        "deployment_count": 0,
        "total_executions": 0,
        "avg_response_time": 0.0,
        "success_rate": 1.0,
        "last_deployment": null
    }',

    -- Constraints
    CONSTRAINT agent_configurations_name_length CHECK (char_length(name) >= 3),
    CONSTRAINT agent_configurations_role_desc_length CHECK (char_length(role_description) >= 10),
    CONSTRAINT agent_configurations_prompt_length CHECK (char_length(system_prompt) >= 50),
    CONSTRAINT agent_configurations_capabilities_type CHECK (jsonb_typeof(capabilities) = 'array'),
    CONSTRAINT agent_configurations_autonomy_type CHECK (jsonb_typeof(autonomy_level) = 'object'),
    CONSTRAINT agent_configurations_metrics_type CHECK (jsonb_typeof(performance_metrics) = 'object')
);

-- Indexes for agent_configurations
CREATE INDEX idx_agent_configurations_user_id ON agent_configurations(user_id);
CREATE INDEX idx_agent_configurations_active ON agent_configurations(user_id, is_active) WHERE is_active = true;
CREATE INDEX idx_agent_configurations_modified ON agent_configurations(last_modified DESC);
CREATE INDEX idx_agent_configurations_name_search ON agent_configurations USING gin(to_tsvector('english', name || ' ' || role_description));

-- =====================================================
-- AGENT TEMPLATES TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS agent_templates (
    template_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    base_prompt TEXT NOT NULL,
    suggested_capabilities JSONB NOT NULL DEFAULT '[]',
    category VARCHAR(100) NOT NULL,
    tags JSONB NOT NULL DEFAULT '[]',
    created_by UUID,
    is_public BOOLEAN DEFAULT false,
    version VARCHAR(20) DEFAULT '1.0.0',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    usage_count INTEGER DEFAULT 0,
    average_rating DECIMAL(3,2) DEFAULT 0.00,
    total_ratings INTEGER DEFAULT 0,
    example_use_cases JSONB DEFAULT '[]',

    -- Constraints
    CONSTRAINT agent_templates_name_length CHECK (char_length(name) >= 3),
    CONSTRAINT agent_templates_desc_length CHECK (char_length(description) >= 10),
    CONSTRAINT agent_templates_prompt_length CHECK (char_length(base_prompt) >= 50),
    CONSTRAINT agent_templates_capabilities_type CHECK (jsonb_typeof(suggested_capabilities) = 'array'),
    CONSTRAINT agent_templates_tags_type CHECK (jsonb_typeof(tags) = 'array'),
    CONSTRAINT agent_templates_examples_type CHECK (jsonb_typeof(example_use_cases) = 'array'),
    CONSTRAINT agent_templates_category_valid CHECK (category IN (
        'research', 'data_analysis', 'synthesis', 'domain_expert',
        'facilitation', 'quality_assurance', 'custom'
    )),
    CONSTRAINT agent_templates_rating_range CHECK (average_rating >= 0.0 AND average_rating <= 5.0),
    CONSTRAINT agent_templates_usage_positive CHECK (usage_count >= 0),
    CONSTRAINT agent_templates_ratings_positive CHECK (total_ratings >= 0)
);

-- Indexes for agent_templates
CREATE INDEX idx_agent_templates_public ON agent_templates(is_public, category) WHERE is_public = true;
CREATE INDEX idx_agent_templates_user ON agent_templates(created_by, updated_at DESC);
CREATE INDEX idx_agent_templates_category ON agent_templates(category, average_rating DESC);
CREATE INDEX idx_agent_templates_rating ON agent_templates(average_rating DESC, usage_count DESC) WHERE is_public = true;
CREATE INDEX idx_agent_templates_usage ON agent_templates(usage_count DESC, average_rating DESC) WHERE is_public = true;
CREATE INDEX idx_agent_templates_search ON agent_templates USING gin(to_tsvector('english', name || ' ' || description));
CREATE INDEX idx_agent_templates_tags ON agent_templates USING gin(tags);

-- =====================================================
-- PROMPT VERSIONS TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS prompt_versions (
    version_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES agent_configurations(agent_id) ON DELETE CASCADE,
    prompt_content TEXT NOT NULL,
    version_number INTEGER NOT NULL,
    change_description TEXT,
    created_by UUID,
    created_at TIMESTAMP DEFAULT NOW(),
    is_active BOOLEAN DEFAULT false,
    performance_metrics JSONB DEFAULT '{
        "test_count": 0,
        "avg_response_time": 0.0,
        "success_rate": 1.0,
        "token_usage": 0
    }',

    -- Constraints
    CONSTRAINT prompt_versions_content_length CHECK (char_length(prompt_content) >= 50),
    CONSTRAINT prompt_versions_version_positive CHECK (version_number > 0),
    CONSTRAINT prompt_versions_metrics_type CHECK (jsonb_typeof(performance_metrics) = 'object'),

    -- Unique constraint to ensure only one active version per agent
    CONSTRAINT prompt_versions_unique_active UNIQUE (agent_id, is_active) DEFERRABLE INITIALLY DEFERRED
);

-- Indexes for prompt_versions
CREATE INDEX idx_prompt_versions_agent_id ON prompt_versions(agent_id, version_number DESC);
CREATE INDEX idx_prompt_versions_active ON prompt_versions(agent_id, is_active) WHERE is_active = true;
CREATE INDEX idx_prompt_versions_created ON prompt_versions(created_at DESC);

-- =====================================================
-- PROMPT TEST RESULTS TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS prompt_test_results (
    test_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version_id UUID NOT NULL REFERENCES prompt_versions(version_id) ON DELETE CASCADE,
    test_input TEXT NOT NULL,
    expected_output TEXT,
    actual_output TEXT NOT NULL,
    test_passed BOOLEAN NOT NULL DEFAULT false,
    execution_time_ms INTEGER NOT NULL DEFAULT 0,
    token_usage INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),

    -- Constraints
    CONSTRAINT prompt_test_results_execution_time_positive CHECK (execution_time_ms >= 0),
    CONSTRAINT prompt_test_results_token_usage_positive CHECK (token_usage >= 0)
);

-- Indexes for prompt_test_results
CREATE INDEX idx_prompt_test_results_version ON prompt_test_results(version_id, created_at DESC);
CREATE INDEX idx_prompt_test_results_passed ON prompt_test_results(version_id, test_passed);

-- =====================================================
-- TEMPLATE RATINGS TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS template_ratings (
    rating_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    template_id UUID NOT NULL REFERENCES agent_templates(template_id) ON DELETE CASCADE,
    user_id UUID NOT NULL,
    rating INTEGER NOT NULL,
    comment TEXT,
    created_at TIMESTAMP DEFAULT NOW(),

    -- Constraints
    CONSTRAINT template_ratings_rating_range CHECK (rating >= 1 AND rating <= 5),
    CONSTRAINT template_ratings_unique_user_template UNIQUE (template_id, user_id)
);

-- Indexes for template_ratings
CREATE INDEX idx_template_ratings_template ON template_ratings(template_id, created_at DESC);
CREATE INDEX idx_template_ratings_user ON template_ratings(user_id, created_at DESC);

-- =====================================================
-- AGENT CAPABILITY DEFINITIONS TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS agent_capability_definitions (
    capability_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT NOT NULL,
    mcp_tool_mapping VARCHAR(255) NOT NULL,
    requires_approval BOOLEAN DEFAULT true,
    estimated_execution_time INTEGER DEFAULT 30,
    category VARCHAR(100) DEFAULT 'general',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),

    -- Constraints
    CONSTRAINT capability_definitions_name_length CHECK (char_length(name) >= 3),
    CONSTRAINT capability_definitions_desc_length CHECK (char_length(description) >= 10),
    CONSTRAINT capability_definitions_execution_time_positive CHECK (estimated_execution_time > 0),
    CONSTRAINT capability_definitions_category_valid CHECK (category IN (
        'general', 'analysis', 'research', 'synthesis', 'collection', 'processing'
    ))
);

-- Indexes for agent_capability_definitions
CREATE INDEX idx_capability_definitions_active ON agent_capability_definitions(is_active, category);
CREATE INDEX idx_capability_definitions_mcp_tool ON agent_capability_definitions(mcp_tool_mapping);

-- =====================================================
-- AGENT DEPLOYMENTS TRACKING TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS agent_deployments (
    deployment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES agent_configurations(agent_id) ON DELETE CASCADE,
    project_id UUID NOT NULL,
    deployment_status VARCHAR(50) DEFAULT 'active',
    deployed_by UUID,
    deployed_at TIMESTAMP DEFAULT NOW(),
    last_activity TIMESTAMP DEFAULT NOW(),
    deployment_metrics JSONB DEFAULT '{
        "total_actions": 0,
        "successful_actions": 0,
        "avg_response_time": 0.0,
        "last_error": null
    }',

    -- Constraints
    CONSTRAINT agent_deployments_status_valid CHECK (deployment_status IN (
        'active', 'inactive', 'error', 'maintenance'
    )),
    CONSTRAINT agent_deployments_metrics_type CHECK (jsonb_typeof(deployment_metrics) = 'object'),

    -- Unique constraint to prevent duplicate active deployments
    CONSTRAINT agent_deployments_unique_active UNIQUE (agent_id, project_id, deployment_status)
    DEFERRABLE INITIALLY DEFERRED
);

-- Indexes for agent_deployments
CREATE INDEX idx_agent_deployments_agent ON agent_deployments(agent_id, deployment_status);
CREATE INDEX idx_agent_deployments_project ON agent_deployments(project_id, deployment_status);
CREATE INDEX idx_agent_deployments_activity ON agent_deployments(last_activity DESC);

-- =====================================================
-- SEED INITIAL CAPABILITY DEFINITIONS
-- =====================================================
INSERT INTO agent_capability_definitions (name, description, mcp_tool_mapping, requires_approval, estimated_execution_time, category) VALUES
    ('SWOT Analysis', 'Strategic analysis of Strengths, Weaknesses, Opportunities, Threats', 'swot_analysis', false, 30, 'analysis'),
    ('ACH Analysis', 'Analysis of Competing Hypotheses for hypothesis testing', 'ach_analysis', true, 60, 'analysis'),
    ('Network Analysis', 'Entity relationship and network structure analysis', 'network_analysis', false, 45, 'analysis'),
    ('Unified Search', 'Comprehensive search across multiple data sources', 'unified_search', false, 15, 'research'),
    ('Create Project', 'Initialize new intelligence projects', 'create_project', true, 10, 'general'),
    ('List Projects', 'Retrieve and display project information', 'list_projects', false, 5, 'general'),
    ('Upload Document', 'Document ingestion and processing', 'upload_document', false, 20, 'collection'),
    ('Bulk Upload', 'Batch document processing and ingestion', 'bulk_upload', true, 60, 'collection')
ON CONFLICT (name) DO NOTHING;

-- =====================================================
-- SEED INITIAL TEMPLATES
-- =====================================================
INSERT INTO agent_templates (
    name, description, base_prompt, suggested_capabilities, category, tags,
    is_public, example_use_cases, created_by
) VALUES
(
    'Intelligence Researcher',
    'Professional intelligence analysis with structured analytic techniques',
    'You are a professional intelligence researcher specialized in structured analytic techniques. Your approach includes:

- Using ACH (Analysis of Competing Hypotheses) to reduce cognitive bias
- Applying SWOT analysis for strategic assessment
- Seeking multiple sources and cross-verification
- Providing confidence levels and methodological justification
- Focusing on actionable intelligence products

Adapt your expertise to the specific research domain and objectives provided by the user.',
    '[
        {"name": "ACH Analysis", "mcp_tool_name": "ach_analysis"},
        {"name": "SWOT Analysis", "mcp_tool_name": "swot_analysis"},
        {"name": "Unified Search", "mcp_tool_name": "unified_search"}
    ]',
    'research',
    '["intelligence", "research", "analysis", "structured-techniques"]',
    true,
    '["Market intelligence research", "Competitive analysis projects", "Policy research and analysis", "Academic research synthesis"]',
    null
),
(
    'Data Analysis Assistant',
    'Statistical analysis and data interpretation specialist',
    'You are a data analysis assistant with expertise in:

- Statistical analysis and hypothesis testing
- Data visualization and pattern recognition
- Network analysis and relationship mapping
- Evidence-based conclusions with uncertainty quantification
- Clear communication of analytical findings

Customize your analytical approach based on the user''s specific data types, research questions, and analytical objectives.',
    '[
        {"name": "Network Analysis", "mcp_tool_name": "network_analysis"},
        {"name": "SWOT Analysis", "mcp_tool_name": "swot_analysis"},
        {"name": "Unified Search", "mcp_tool_name": "unified_search"}
    ]',
    'data_analysis',
    '["data-analysis", "statistics", "visualization", "patterns"]',
    true,
    '["Social network analysis", "Business data exploration", "Research data interpretation", "Performance metrics analysis"]',
    null
),
(
    'Research Assistant',
    'General-purpose research and analysis agent',
    'You are a versatile research assistant capable of adapting to various research domains and methodologies. Your core competencies include:

- Systematic information gathering and verification
- Critical analysis and evidence evaluation
- Research methodology application
- Clear documentation and reporting
- Collaborative research support

Adapt your approach, language, and analytical methods to match the user''s specific research field, objectives, and preferred methodologies.',
    '[
        {"name": "Unified Search", "mcp_tool_name": "unified_search"},
        {"name": "SWOT Analysis", "mcp_tool_name": "swot_analysis"},
        {"name": "Create Project", "mcp_tool_name": "create_project"}
    ]',
    'research',
    '["research", "assistant", "general-purpose", "adaptable"]',
    true,
    '["Academic research support", "Business research projects", "Personal investigation tasks", "Learning and exploration"]',
    null
)
ON CONFLICT DO NOTHING;

-- =====================================================
-- CREATE TRIGGERS FOR AUTOMATIC UPDATES
-- =====================================================

-- Trigger to update agent_configurations.last_modified
CREATE OR REPLACE FUNCTION update_modified_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_modified = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_agent_configurations_modified
    BEFORE UPDATE ON agent_configurations
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_timestamp();

CREATE TRIGGER update_agent_templates_modified
    BEFORE UPDATE ON agent_templates
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_timestamp();

-- Trigger to update template ratings when new rating is added
CREATE OR REPLACE FUNCTION update_template_rating_stats()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' OR TG_OP = 'UPDATE' THEN
        UPDATE agent_templates
        SET
            average_rating = (
                SELECT COALESCE(AVG(rating), 0.0)
                FROM template_ratings
                WHERE template_id = NEW.template_id
            ),
            total_ratings = (
                SELECT COUNT(*)
                FROM template_ratings
                WHERE template_id = NEW.template_id
            ),
            updated_at = NOW()
        WHERE template_id = NEW.template_id;
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE agent_templates
        SET
            average_rating = (
                SELECT COALESCE(AVG(rating), 0.0)
                FROM template_ratings
                WHERE template_id = OLD.template_id
            ),
            total_ratings = (
                SELECT COUNT(*)
                FROM template_ratings
                WHERE template_id = OLD.template_id
            ),
            updated_at = NOW()
        WHERE template_id = OLD.template_id;
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_template_rating_stats_trigger
    AFTER INSERT OR UPDATE OR DELETE ON template_ratings
    FOR EACH ROW
    EXECUTE FUNCTION update_template_rating_stats();

-- =====================================================
-- CREATE VIEWS FOR COMMON QUERIES
-- =====================================================

-- View for active agent configurations with latest prompt
CREATE OR REPLACE VIEW active_agent_configurations AS
SELECT
    ac.*,
    pv.prompt_content as current_prompt,
    pv.version_number as current_prompt_version,
    pv.version_id as current_prompt_version_id
FROM agent_configurations ac
LEFT JOIN prompt_versions pv ON ac.agent_id = pv.agent_id AND pv.is_active = true
WHERE ac.is_active = true;

-- View for public templates with rating stats
CREATE OR REPLACE VIEW public_templates_with_stats AS
SELECT
    at.*,
    COALESCE(recent_usage.recent_usage_count, 0) as recent_usage_count
FROM agent_templates at
LEFT JOIN (
    SELECT
        template_id,
        COUNT(*) as recent_usage_count
    FROM template_ratings
    WHERE created_at >= NOW() - INTERVAL '30 days'
    GROUP BY template_id
) recent_usage ON at.template_id = recent_usage.template_id
WHERE at.is_public = true;

-- View for agent deployment summary
CREATE OR REPLACE VIEW agent_deployment_summary AS
SELECT
    ac.agent_id,
    ac.name,
    ac.user_id,
    COUNT(ad.deployment_id) as active_deployments,
    MAX(ad.last_activity) as last_activity,
    AVG((ad.deployment_metrics->>'total_actions')::integer) as avg_total_actions
FROM agent_configurations ac
LEFT JOIN agent_deployments ad ON ac.agent_id = ad.agent_id AND ad.deployment_status = 'active'
WHERE ac.is_active = true
GROUP BY ac.agent_id, ac.name, ac.user_id;

-- =====================================================
-- GRANT PERMISSIONS
-- =====================================================

-- Note: In a real implementation, you would grant specific permissions
-- to application users. This is a template for the required permissions.

/*
GRANT SELECT, INSERT, UPDATE, DELETE ON agent_configurations TO app_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON agent_templates TO app_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON prompt_versions TO app_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON prompt_test_results TO app_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON template_ratings TO app_user;
GRANT SELECT ON agent_capability_definitions TO app_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON agent_deployments TO app_user;

GRANT SELECT ON active_agent_configurations TO app_user;
GRANT SELECT ON public_templates_with_stats TO app_user;
GRANT SELECT ON agent_deployment_summary TO app_user;

GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO app_user;
*/

-- =====================================================
-- MIGRATION COMPLETE
-- =====================================================

-- Log migration completion
INSERT INTO schema_migrations (version, applied_at) VALUES ('004', NOW())
ON CONFLICT (version) DO NOTHING;
