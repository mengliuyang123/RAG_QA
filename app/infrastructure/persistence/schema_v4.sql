-- 中文分词由应用统一执行；生成列和 GIN 与文本块写入/删除同事务维护。
ALTER TABLE vector_chunks ADD COLUMN search_text TEXT;
ALTER TABLE vector_chunks ADD COLUMN search_vector tsvector
    GENERATED ALWAYS AS (to_tsvector('pg_catalog.simple'::regconfig, search_text)) STORED;
