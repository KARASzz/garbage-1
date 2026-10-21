PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS categories (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    description TEXT NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS fragments (
    id TEXT PRIMARY KEY,
    category_id TEXT NOT NULL REFERENCES categories(id),
    text TEXT NOT NULL CHECK (length(trim(text)) > 0),
    weight REAL NOT NULL CHECK (weight > 0)
) STRICT;
CREATE INDEX IF NOT EXISTS fragments_category ON fragments(category_id);

CREATE TABLE IF NOT EXISTS tags (id TEXT PRIMARY KEY) STRICT;
CREATE TABLE IF NOT EXISTS fragment_tags (
    fragment_id TEXT NOT NULL REFERENCES fragments(id) ON DELETE CASCADE,
    tag_id TEXT NOT NULL REFERENCES tags(id),
    PRIMARY KEY (fragment_id, tag_id)
) STRICT;
CREATE INDEX IF NOT EXISTS fragment_tags_tag ON fragment_tags(tag_id, fragment_id);

CREATE TABLE IF NOT EXISTS templates (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    theme TEXT NOT NULL,
    weight REAL NOT NULL CHECK (weight > 0)
) STRICT;

CREATE TABLE IF NOT EXISTS template_slots (
    template_id TEXT NOT NULL REFERENCES templates(id) ON DELETE CASCADE,
    position INTEGER NOT NULL CHECK (position >= 0),
    kind TEXT NOT NULL CHECK (kind IN ('literal', 'slot')),
    slot_name TEXT,
    literal TEXT,
    category_id TEXT REFERENCES categories(id),
    optional_probability REAL CHECK (optional_probability BETWEEN 0 AND 1),
    prefix TEXT NOT NULL DEFAULT '',
    suffix TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (template_id, position),
    CHECK ((kind = 'literal' AND literal IS NOT NULL AND category_id IS NULL)
        OR (kind = 'slot' AND category_id IS NOT NULL AND slot_name IS NOT NULL))
) STRICT;

CREATE TABLE IF NOT EXISTS compatibility_rules (
    template_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    tag_id TEXT NOT NULL REFERENCES tags(id),
    mode TEXT NOT NULL CHECK (mode IN ('all', 'any', 'exclude')),
    PRIMARY KEY (template_id, position, tag_id, mode),
    FOREIGN KEY (template_id, position) REFERENCES template_slots(template_id, position) ON DELETE CASCADE
) STRICT;
