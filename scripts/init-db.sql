-- Enable the pgvector extension for vector similarity search.
-- This script runs automatically when the PostgreSQL container is first
-- initialised via Docker's docker-entrypoint-initdb.d mechanism.
CREATE EXTENSION IF NOT EXISTS vector;
