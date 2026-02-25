-- Align embedding column with 768-dim (nomic-embed-text).
-- Drop vector index, resize column (existing values set to NULL), recreate index.
DO $$
DECLARE
  rec RECORD;
BEGIN
  DROP INDEX IF EXISTS documents_embedding_idx;
  DROP INDEX IF EXISTS documents_embedding_vector_cosine_ops_idx;
  FOR rec IN (SELECT indexname FROM pg_indexes WHERE schemaname = 'public' AND tablename = 'documents' AND indexdef LIKE '%embedding%')
  LOOP
    EXECUTE format('DROP INDEX IF EXISTS %I', rec.indexname);
  END LOOP;
END $$;
ALTER TABLE documents ALTER COLUMN embedding TYPE vector(768) USING NULL;
CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
