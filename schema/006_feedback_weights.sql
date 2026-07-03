-- Track how often each episode is retrieved so recall() can learn what matters.
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS retrieval_count INTEGER NOT NULL DEFAULT 0;
CREATE INDEX IF NOT EXISTS idx_episodes_retrieval_count ON episodes (retrieval_count DESC);
