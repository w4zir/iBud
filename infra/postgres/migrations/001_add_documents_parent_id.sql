-- Add any missing columns to documents (for DBs created before full schema)
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'documents' AND column_name = 'parent_id') THEN
    ALTER TABLE documents ADD COLUMN parent_id UUID REFERENCES documents(id) ON DELETE SET NULL;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'documents' AND column_name = 'source') THEN
    ALTER TABLE documents ADD COLUMN source VARCHAR(50);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'documents' AND column_name = 'doc_tier') THEN
    ALTER TABLE documents ADD COLUMN doc_tier INTEGER DEFAULT 1;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'documents' AND column_name = 'category') THEN
    ALTER TABLE documents ADD COLUMN category VARCHAR(100);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'documents' AND column_name = 'source_id') THEN
    ALTER TABLE documents ADD COLUMN source_id VARCHAR(200);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'documents' AND column_name = 'metadata') THEN
    ALTER TABLE documents ADD COLUMN metadata JSONB;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'documents' AND column_name = 'created_at') THEN
    ALTER TABLE documents ADD COLUMN created_at TIMESTAMPTZ DEFAULT NOW();
  END IF;
END $$;

-- Ensure useful indexes exist (idempotent: CREATE INDEX IF NOT EXISTS in PG 9.5+)
CREATE INDEX IF NOT EXISTS idx_documents_doc_tier ON documents (doc_tier);
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents (source);
