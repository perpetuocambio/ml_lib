-- SQLite Schema for LoRA Model Repository
-- Clean architecture persistence layer

-- Main LoRAs table
CREATE TABLE IF NOT EXISTS loras (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    path TEXT NOT NULL,
    base_model TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    download_count INTEGER NOT NULL DEFAULT 0,
    rating REAL NOT NULL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CHECK (weight >= 0.0 AND weight <= 2.0),
    CHECK (rating >= 0.0 AND rating <= 5.0),
    CHECK (download_count >= 0)
);

-- Trigger words for LoRAs (many-to-many)
CREATE TABLE IF NOT EXISTS lora_trigger_words (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lora_id INTEGER NOT NULL,
    trigger_word TEXT NOT NULL,

    FOREIGN KEY (lora_id) REFERENCES loras(id) ON DELETE CASCADE,
    UNIQUE(lora_id, trigger_word)
);

-- Tags for LoRAs (many-to-many)
CREATE TABLE IF NOT EXISTS lora_tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lora_id INTEGER NOT NULL,
    tag TEXT NOT NULL,

    FOREIGN KEY (lora_id) REFERENCES loras(id) ON DELETE CASCADE,
    UNIQUE(lora_id, tag)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_loras_name ON loras(name);
CREATE INDEX IF NOT EXISTS idx_loras_base_model ON loras(base_model);
CREATE INDEX IF NOT EXISTS idx_loras_rating ON loras(rating DESC);
CREATE INDEX IF NOT EXISTS idx_loras_downloads ON loras(download_count DESC);
CREATE INDEX IF NOT EXISTS idx_trigger_words_lora ON lora_trigger_words(lora_id);
CREATE INDEX IF NOT EXISTS idx_trigger_words_word ON lora_trigger_words(trigger_word);
CREATE INDEX IF NOT EXISTS idx_tags_lora ON lora_tags(lora_id);
CREATE INDEX IF NOT EXISTS idx_tags_tag ON lora_tags(tag);

-- Trigger to update updated_at timestamp
CREATE TRIGGER IF NOT EXISTS update_loras_timestamp
AFTER UPDATE ON loras
BEGIN
    UPDATE loras SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;
