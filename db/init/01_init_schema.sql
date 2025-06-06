-- OmniBio Database Schema
-- This script initializes the database with all required tables

-- Create extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Jobs table - tracks analysis jobs
CREATE TABLE jobs (
    id SERIAL PRIMARY KEY,
    analysis_id UUID UNIQUE NOT NULL DEFAULT uuid_generate_v4(),
    project_name VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    analysis_types TEXT[] NOT NULL,
    file_ids TEXT[] NOT NULL,
    group_column VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    progress DECIMAL(5,2) DEFAULT 0.0,
    message TEXT DEFAULT 'Queued',
    error_message TEXT,
    results_path TEXT,
    results_data JSONB,
    created_by UUID,
    
    CONSTRAINT jobs_status_check CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    CONSTRAINT jobs_progress_check CHECK (progress >= 0.0 AND progress <= 100.0)
);

-- Files table - tracks uploaded files
CREATE TABLE files (
    id SERIAL PRIMARY KEY,
    file_id VARCHAR(255) UNIQUE NOT NULL,
    original_filename VARCHAR(255) NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    file_path TEXT NOT NULL,
    size_bytes BIGINT NOT NULL,
    checksum VARCHAR(64),
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    uploaded_by UUID,
    status VARCHAR(50) DEFAULT 'uploaded',
    file_metadata JSONB,
    
    CONSTRAINT files_status_check CHECK (status IN ('uploaded', 'processing', 'processed', 'error', 'deleted'))
);

-- Users table - simple user management for API access
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    api_key VARCHAR(255) UNIQUE NOT NULL,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE,
    permissions JSONB DEFAULT '{"read": true, "write": true, "admin": false}'::jsonb
);

-- API keys table - for managing multiple API keys per user
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    key_name VARCHAR(100) NOT NULL,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    permissions JSONB DEFAULT '{"read": true, "write": true}'::jsonb,
    expires_at TIMESTAMP WITH TIME ZONE,
    last_used TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT true
);

-- Job files junction table - many-to-many relationship
CREATE TABLE job_files (
    job_id INTEGER REFERENCES jobs(id) ON DELETE CASCADE,
    file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
    PRIMARY KEY (job_id, file_id)
);

-- Create indexes for better performance
CREATE INDEX idx_jobs_analysis_id ON jobs(analysis_id);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_created_at ON jobs(created_at);
CREATE INDEX idx_jobs_created_by ON jobs(created_by);

CREATE INDEX idx_files_file_id ON files(file_id);
CREATE INDEX idx_files_uploaded_by ON files(uploaded_by);
CREATE INDEX idx_files_uploaded_at ON files(uploaded_at);

CREATE INDEX idx_users_api_key ON users(api_key);
CREATE INDEX idx_users_email ON users(email);

CREATE INDEX idx_api_keys_key_hash ON api_keys(key_hash);
CREATE INDEX idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX idx_api_keys_expires_at ON api_keys(expires_at);

-- Insert a default user for development
INSERT INTO users (username, email, api_key, permissions) VALUES (
    'dev_user',
    'dev@omnibio.com',
    'omnibio-dev-key-12345',
    '{"read": true, "write": true, "admin": true}'::jsonb
);

-- Insert some sample API keys
INSERT INTO api_keys (key_hash, key_name, user_id, permissions) VALUES (
    'sha256_hash_of_omnibio-dev-key-12345',
    'Development Key',
    (SELECT id FROM users WHERE username = 'dev_user'),
    '{"read": true, "write": true, "admin": true}'::jsonb
);

COMMENT ON TABLE jobs IS 'Analysis jobs submitted through the API';
COMMENT ON TABLE files IS 'Uploaded files for analysis';
COMMENT ON TABLE users IS 'Users with API access';
COMMENT ON TABLE api_keys IS 'API keys for authentication';
COMMENT ON TABLE job_files IS 'Junction table linking jobs to their input files'; 