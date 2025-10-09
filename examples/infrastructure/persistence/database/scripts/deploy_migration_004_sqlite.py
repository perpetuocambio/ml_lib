#!/usr/bin/env python3
"""
SQLite-compatible deployment for Migration 004: Dynamic Agent Configuration Tables
EPIC-009: Dynamic Agent Configuration System
"""

import json
import sqlite3
import uuid
from sqlite3 import Connection


def create_sqlite_migration() -> str:
    """Create SQLite-compatible version of Migration 004"""

    migration_sql = """
-- Migration 004: Dynamic Agent Configuration Tables (SQLite version)
-- EPIC-009: Dynamic Agent Configuration System

-- =====================================================
-- AGENT CONFIGURATIONS TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS agent_configurations (
    agent_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    role_description TEXT NOT NULL,
    system_prompt TEXT NOT NULL,
    capabilities TEXT NOT NULL DEFAULT '[]',
    autonomy_level TEXT NOT NULL DEFAULT '{}',
    knowledge_context TEXT,
    is_active INTEGER DEFAULT 1,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    last_modified TEXT DEFAULT CURRENT_TIMESTAMP,
    performance_metrics TEXT DEFAULT '{"deployment_count": 0, "total_executions": 0, "avg_response_time": 0.0, "success_rate": 1.0, "last_deployment": null}',

    -- Constraints
    CHECK (length(name) >= 3),
    CHECK (length(role_description) >= 10),
    CHECK (length(system_prompt) >= 50)
);

-- Indexes for agent_configurations
CREATE INDEX IF NOT EXISTS idx_agent_configurations_user_id ON agent_configurations(user_id);
CREATE INDEX IF NOT EXISTS idx_agent_configurations_active ON agent_configurations(user_id, is_active);
CREATE INDEX IF NOT EXISTS idx_agent_configurations_modified ON agent_configurations(last_modified DESC);

-- =====================================================
-- AGENT TEMPLATES TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS agent_templates (
    template_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    role_archetype TEXT NOT NULL,
    base_system_prompt TEXT NOT NULL,
    default_capabilities TEXT NOT NULL DEFAULT '[]',
    suggested_autonomy TEXT NOT NULL DEFAULT '{}',
    usage_examples TEXT,
    category TEXT DEFAULT 'general',
    complexity_level TEXT DEFAULT 'intermediate',
    is_official INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    usage_count INTEGER DEFAULT 0,
    avg_rating REAL DEFAULT 0.0,

    -- Constraints
    CHECK (length(name) >= 3),
    CHECK (length(description) >= 20),
    CHECK (length(base_system_prompt) >= 100),
    CHECK (category IN ('general', 'research', 'analysis', 'technical', 'creative', 'specialized')),
    CHECK (complexity_level IN ('beginner', 'intermediate', 'advanced', 'expert')),
    CHECK (avg_rating >= 0.0 AND avg_rating <= 5.0)
);

-- Indexes for agent_templates
CREATE INDEX IF NOT EXISTS idx_agent_templates_category ON agent_templates(category);
CREATE INDEX IF NOT EXISTS idx_agent_templates_complexity ON agent_templates(complexity_level);
CREATE INDEX IF NOT EXISTS idx_agent_templates_official ON agent_templates(is_official);
CREATE INDEX IF NOT EXISTS idx_agent_templates_rating ON agent_templates(avg_rating DESC);

-- =====================================================
-- AGENT DEPLOYMENT HISTORY TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS agent_deployment_history (
    deployment_id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    project_id TEXT,
    deployed_at TEXT DEFAULT CURRENT_TIMESTAMP,
    deployment_duration INTEGER,
    execution_count INTEGER DEFAULT 0,
    status TEXT DEFAULT 'active',
    performance_snapshot TEXT DEFAULT '{}',

    -- Constraints
    CHECK (status IN ('active', 'paused', 'completed', 'terminated', 'error')),
    CHECK (execution_count >= 0),

    FOREIGN KEY (agent_id) REFERENCES agent_configurations(agent_id) ON DELETE CASCADE
);

-- Indexes for deployment history
CREATE INDEX IF NOT EXISTS idx_deployment_agent_id ON agent_deployment_history(agent_id);
CREATE INDEX IF NOT EXISTS idx_deployment_project_id ON agent_deployment_history(project_id);
CREATE INDEX IF NOT EXISTS idx_deployment_status ON agent_deployment_history(status);
CREATE INDEX IF NOT EXISTS idx_deployment_date ON agent_deployment_history(deployed_at DESC);

-- =====================================================
-- AGENT EXECUTION LOGS TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS agent_execution_logs (
    execution_id TEXT PRIMARY KEY,
    deployment_id TEXT NOT NULL,
    executed_at TEXT DEFAULT CURRENT_TIMESTAMP,
    execution_type TEXT NOT NULL,
    input_data TEXT,
    output_data TEXT,
    execution_time_ms INTEGER,
    success INTEGER DEFAULT 1,
    error_details TEXT,

    -- Constraints
    CHECK (execution_type IN ('analysis', 'synthesis', 'data_collection', 'report_generation', 'custom')),
    CHECK (execution_time_ms >= 0),

    FOREIGN KEY (deployment_id) REFERENCES agent_deployment_history(deployment_id) ON DELETE CASCADE
);

-- Indexes for execution logs
CREATE INDEX IF NOT EXISTS idx_execution_deployment ON agent_execution_logs(deployment_id);
CREATE INDEX IF NOT EXISTS idx_execution_type ON agent_execution_logs(execution_type);
CREATE INDEX IF NOT EXISTS idx_execution_success ON agent_execution_logs(success);
CREATE INDEX IF NOT EXISTS idx_execution_time ON agent_execution_logs(executed_at DESC);

-- =====================================================
-- AGENT RATINGS TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS agent_ratings (
    rating_id TEXT PRIMARY KEY,
    agent_id TEXT,
    template_id TEXT,
    user_id TEXT NOT NULL,
    rating REAL NOT NULL,
    review_text TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CHECK (rating >= 1.0 AND rating <= 5.0),
    CHECK ((agent_id IS NOT NULL AND template_id IS NULL) OR (agent_id IS NULL AND template_id IS NOT NULL)),

    FOREIGN KEY (agent_id) REFERENCES agent_configurations(agent_id) ON DELETE CASCADE,
    FOREIGN KEY (template_id) REFERENCES agent_templates(template_id) ON DELETE CASCADE
);

-- Indexes for ratings
CREATE INDEX IF NOT EXISTS idx_ratings_agent ON agent_ratings(agent_id);
CREATE INDEX IF NOT EXISTS idx_ratings_template ON agent_ratings(template_id);
CREATE INDEX IF NOT EXISTS idx_ratings_user ON agent_ratings(user_id);
CREATE INDEX IF NOT EXISTS idx_ratings_value ON agent_ratings(rating DESC);

-- =====================================================
-- AGENT CAPABILITIES TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS agent_capabilities (
    capability_id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    description TEXT NOT NULL,
    category TEXT DEFAULT 'general',
    implementation_complexity TEXT DEFAULT 'medium',
    is_core INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CHECK (length(name) >= 2),
    CHECK (length(description) >= 10),
    CHECK (category IN ('core', 'analysis', 'data', 'communication', 'specialized', 'experimental')),
    CHECK (implementation_complexity IN ('low', 'medium', 'high', 'very_high'))
);

-- Indexes for capabilities
CREATE INDEX IF NOT EXISTS idx_capabilities_category ON agent_capabilities(category);
CREATE INDEX IF NOT EXISTS idx_capabilities_complexity ON agent_capabilities(implementation_complexity);
CREATE INDEX IF NOT EXISTS idx_capabilities_core ON agent_capabilities(is_core);

-- =====================================================
-- AGENT AUTONOMY LEVELS TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS agent_autonomy_levels (
    level_id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    description TEXT NOT NULL,
    approval_threshold REAL DEFAULT 0.5,
    auto_execute_limit INTEGER DEFAULT 10,
    risk_assessment_required INTEGER DEFAULT 1,
    escalation_rules TEXT DEFAULT '{}',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CHECK (length(name) >= 3),
    CHECK (length(description) >= 15),
    CHECK (approval_threshold >= 0.0 AND approval_threshold <= 1.0),
    CHECK (auto_execute_limit >= 0)
);

-- Indexes for autonomy levels
CREATE INDEX IF NOT EXISTS idx_autonomy_threshold ON agent_autonomy_levels(approval_threshold);
CREATE INDEX IF NOT EXISTS idx_autonomy_limit ON agent_autonomy_levels(auto_execute_limit);
"""

    return migration_sql


def deploy_migration() -> bool:
    """Deploy Migration 004 to SQLite database with seed data"""
    db_path = "pyintelcivil.db"

    try:
        with sqlite3.connect(db_path) as conn:
            # Deploy schema
            migration_sql = create_sqlite_migration()
            conn.executescript(migration_sql)
            print("✅ Migration 004 schema deployed successfully")

            # Insert seed data
            insert_seed_data(conn)
            print("✅ Seed data inserted successfully")

            # Verify tables
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%agent%'"
            )
            tables = cursor.fetchall()

            print("📋 Agent tables created:")
            for table in tables:
                print(f"  - {table[0]}")

            return True

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False


def insert_seed_data(conn: Connection) -> None:
    """Insert seed data for agent templates and capabilities"""

    # Generate UUIDs for seed data
    template_ids = {
        "researcher": str(uuid.uuid4()),
        "analyst": str(uuid.uuid4()),
        "synthesizer": str(uuid.uuid4()),
    }

    capability_ids = {
        "data_analysis": str(uuid.uuid4()),
        "web_research": str(uuid.uuid4()),
        "synthesis": str(uuid.uuid4()),
    }

    autonomy_ids = {
        "supervised": str(uuid.uuid4()),
        "semi_autonomous": str(uuid.uuid4()),
        "autonomous": str(uuid.uuid4()),
    }

    # Insert capabilities
    capabilities = [
        (
            capability_ids["data_analysis"],
            "Data Analysis",
            "Advanced statistical and quantitative analysis capabilities",
            "analysis",
            "medium",
            1,
        ),
        (
            capability_ids["web_research"],
            "Web Research",
            "Systematic web scraping and information gathering",
            "data",
            "medium",
            1,
        ),
        (
            capability_ids["synthesis"],
            "Synthesis",
            "Cross-domain synthesis and insight generation",
            "analysis",
            "high",
            1,
        ),
    ]

    conn.executemany(
        """
        INSERT INTO agent_capabilities (capability_id, name, description, category, implementation_complexity, is_core)
        VALUES (?, ?, ?, ?, ?, ?)
    """,
        capabilities,
    )

    # Insert autonomy levels
    autonomy_levels = [
        (
            autonomy_ids["supervised"],
            "Supervised",
            "Requires approval for all actions",
            1.0,
            0,
            1,
            '{"escalate_all": true}',
        ),
        (
            autonomy_ids["semi_autonomous"],
            "Semi-Autonomous",
            "Can auto-execute routine actions",
            0.7,
            5,
            1,
            '{"routine_threshold": 0.3}',
        ),
        (
            autonomy_ids["autonomous"],
            "Autonomous",
            "High autonomy with strategic oversight",
            0.3,
            20,
            0,
            '{"strategic_only": true}',
        ),
    ]

    conn.executemany(
        """
        INSERT INTO agent_autonomy_levels (level_id, name, description, approval_threshold, auto_execute_limit, risk_assessment_required, escalation_rules)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
        autonomy_levels,
    )

    # Insert agent templates
    templates = [
        (
            template_ids["researcher"],
            "Research Specialist",
            "Specialized agent for systematic research and data collection",
            "researcher",
            "You are a research specialist focused on systematic data collection and analysis. Your role is to gather comprehensive information on assigned topics using rigorous methodological approaches.",
            json.dumps(["web_research", "data_analysis"]),
            json.dumps({"level": "semi_autonomous", "auto_approve_research": True}),
            "Best for: Literature reviews, market research, competitive analysis",
            "research",
            "intermediate",
            1,
            0,
            4.2,
        ),
        (
            template_ids["analyst"],
            "Intelligence Analyst",
            "Expert in structured analysis techniques and synthesis",
            "analyst",
            "You are an intelligence analyst specializing in structured analysis techniques. Apply ACH, SWOT, Network Analysis and other methodologies to derive actionable insights.",
            json.dumps(["data_analysis", "synthesis"]),
            json.dumps({"level": "supervised", "requires_validation": True}),
            "Best for: ACH analysis, threat assessment, strategic planning",
            "analysis",
            "advanced",
            1,
            0,
            4.5,
        ),
        (
            template_ids["synthesizer"],
            "Cross-Domain Synthesizer",
            "Integrates insights across multiple domains and sources",
            "synthesizer",
            "You are a cross-domain synthesizer who excels at connecting insights from disparate sources and domains. Create coherent narratives from complex multi-source data.",
            json.dumps(["synthesis", "data_analysis"]),
            json.dumps({"level": "autonomous", "creativity_mode": True}),
            "Best for: Final reports, executive summaries, strategic recommendations",
            "creative",
            "expert",
            1,
            0,
            4.0,
        ),
    ]

    conn.executemany(
        """
        INSERT INTO agent_templates (template_id, name, description, role_archetype, base_system_prompt,
                                   default_capabilities, suggested_autonomy, usage_examples, category,
                                   complexity_level, is_official, usage_count, avg_rating)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        templates,
    )


if __name__ == "__main__":
    success = deploy_migration()
    exit(0 if success else 1)
