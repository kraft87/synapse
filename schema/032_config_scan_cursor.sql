-- 032: config-lane scan watermark (singleton). The dream config proposer scans only sessions
-- TOUCHED since last_scan_at — same incremental model as the skills lane's skill_scan_cursor.
CREATE TABLE IF NOT EXISTS config_lane.config_scan_cursor (
    id           boolean PRIMARY KEY DEFAULT true CHECK (id),
    last_scan_at timestamptz
);
INSERT INTO config_lane.config_scan_cursor (id, last_scan_at)
    VALUES (true, NULL) ON CONFLICT (id) DO NOTHING;
