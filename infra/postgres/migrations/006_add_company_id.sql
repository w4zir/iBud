-- 006_add_company_id.sql
-- Add company_id columns to core tables for multi-tenant scoping.

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS company_id VARCHAR(100);

ALTER TABLE sessions
    ADD COLUMN IF NOT EXISTS company_id VARCHAR(100);

ALTER TABLE orders
    ADD COLUMN IF NOT EXISTS company_id VARCHAR(100);

-- Basic indexes to keep scoped queries efficient.
CREATE INDEX IF NOT EXISTS idx_documents_company_id ON documents (company_id);
CREATE INDEX IF NOT EXISTS idx_orders_company_id ON orders (company_id);

