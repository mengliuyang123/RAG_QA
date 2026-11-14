-- v4 已用于全文检索；仅新增字段，旧正式轮次默认空数组。
ALTER TABLE conversation_turns
    ADD COLUMN tool_calls JSONB NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(tool_calls) = 'array'),
    ADD COLUMN data_blocks JSONB NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(data_blocks) = 'array');
