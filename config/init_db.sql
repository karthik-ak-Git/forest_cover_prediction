-- Initialize database for Forest Cover Prediction

-- Create database (if running manually)
-- CREATE DATABASE forest_cover_db;

-- Create predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    request_id VARCHAR(100) UNIQUE NOT NULL,
    user_id VARCHAR(100),
    elevation INTEGER,
    aspect INTEGER,
    slope INTEGER,
    predicted_class INTEGER NOT NULL,
    cover_type VARCHAR(50) NOT NULL,
    confidence FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user_id (user_id),
    INDEX idx_created_at (created_at),
    INDEX idx_predicted_class (predicted_class)
);

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    is_superuser BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    INDEX idx_username (username),
    INDEX idx_email (email)
);

-- Create model_metrics table
CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    accuracy FLOAT,
    precision_avg FLOAT,
    recall_avg FLOAT,
    f1_score_avg FLOAT,
    deployed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    INDEX idx_model_name (model_name),
    INDEX idx_deployed_at (deployed_at)
);

-- Create drift_monitoring table
CREATE TABLE IF NOT EXISTS drift_monitoring (
    id SERIAL PRIMARY KEY,
    feature_name VARCHAR(100) NOT NULL,
    baseline_mean FLOAT,
    current_mean FLOAT,
    baseline_std FLOAT,
    current_std FLOAT,
    drift_score FLOAT,
    is_drifting BOOLEAN DEFAULT FALSE,
    checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_feature_name (feature_name),
    INDEX idx_checked_at (checked_at)
);

-- Create MLflow database
CREATE DATABASE IF NOT EXISTS mlflow_db;

-- Insert sample data for testing
INSERT INTO users (username, email, hashed_password, is_superuser) 
VALUES ('admin', 'admin@forestcover.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYzS1TYUJQu', TRUE)
ON CONFLICT (username) DO NOTHING;

-- Create view for prediction statistics
CREATE OR REPLACE VIEW prediction_stats AS
SELECT 
    DATE(created_at) as date,
    cover_type,
    COUNT(*) as prediction_count,
    AVG(confidence) as avg_confidence
FROM predictions
GROUP BY DATE(created_at), cover_type
ORDER BY date DESC, cover_type;

-- Grant permissions (adjust as needed)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO forest_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO forest_user;
