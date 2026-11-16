ALTER TABLE conversations ADD COLUMN IF NOT EXISTS sequence_id BIGINT GENERATED ALWAYS AS IDENTITY UNIQUE;
CREATE TABLE lane_manifest (run_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
CREATE TABLE lane_observations (
    run_id TEXT NOT NULL REFERENCES lane_manifest(run_id) ON DELETE CASCADE,
    episode_id TEXT NOT NULL, policy TEXT NOT NULL, turn_index INTEGER NOT NULL,
    payload TEXT NOT NULL, expires_at DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (run_id, episode_id, policy, turn_index)
);
CREATE TABLE lane_scores (
    run_id TEXT NOT NULL, episode_id TEXT NOT NULL, policy TEXT NOT NULL, turn_index INTEGER NOT NULL,
    payload TEXT NOT NULL, expires_at DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (run_id, episode_id, policy, turn_index),
    FOREIGN KEY (run_id, episode_id, policy, turn_index) REFERENCES lane_observations ON DELETE CASCADE
);
