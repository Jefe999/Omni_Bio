"""
Configuration Management for OmniBio
Uses Pydantic BaseSettings for environment-aware configuration.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from functools import lru_cache

from pydantic import validator, Field
from pydantic_settings import BaseSettings
import yaml


class DatabaseConfig(BaseSettings):
    """Database configuration settings."""
    
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5433, description="Database port")
    name: str = Field(default="omnibio", description="Database name")
    user: str = Field(default="omnibio", description="Database user")
    password: str = Field(default="omnibio123", description="Database password")
    url: Optional[str] = Field(default=None, description="Complete database URL (overrides other settings)")
    
    @validator('url', pre=True, always=True)
    def assemble_db_url(cls, v, values):
        if isinstance(v, str):
            return v
        # Use asyncpg for async PostgreSQL connections
        return f"postgresql+asyncpg://{values.get('user')}:{values.get('password')}@{values.get('host')}:{values.get('port')}/{values.get('name')}"
    
    class Config:
        env_prefix = "DB_"


class StorageConfig(BaseSettings):
    """Storage configuration for files and artifacts."""
    
    upload_dir: Path = Field(default=Path("uploads"), description="Directory for uploaded files")
    results_dir: Path = Field(default=Path("results"), description="Directory for analysis results")
    packages_dir: Path = Field(default=Path("packages"), description="Directory for packaged artifacts")
    temp_dir: Path = Field(default=Path("temp"), description="Temporary directory")
    max_file_size_mb: int = Field(default=500, description="Maximum file size in MB")
    
    # MinIO/S3 configuration
    use_s3: bool = Field(default=False, description="Use S3-compatible storage")
    s3_endpoint: Optional[str] = Field(default=None, description="S3 endpoint URL")
    s3_access_key: Optional[str] = Field(default=None, description="S3 access key")
    s3_secret_key: Optional[str] = Field(default=None, description="S3 secret key")
    s3_bucket: str = Field(default="omnibio", description="S3 bucket name")
    
    class Config:
        env_prefix = "STORAGE_"


class ProcessingConfig(BaseSettings):
    """Configuration for data processing pipelines."""
    
    # Feature extraction
    default_mass_error_ppm: float = Field(default=5.0, description="Default mass error tolerance (ppm)")
    default_rt_tolerance_min: float = Field(default=0.1, description="Default RT tolerance (minutes)")
    max_features: int = Field(default=50000, description="Maximum number of features to process")
    
    # Statistical analysis
    default_p_threshold: float = Field(default=0.05, description="Default p-value threshold")
    default_fc_threshold: float = Field(default=1.0, description="Default fold-change threshold")
    fdr_method: str = Field(default="fdr_bh", description="FDR correction method")
    
    # Machine learning
    default_cv_folds: int = Field(default=5, description="Default cross-validation folds")
    random_state: int = Field(default=42, description="Random seed for reproducibility")
    
    class Config:
        env_prefix = "PROCESSING_"


class SecurityConfig(BaseSettings):
    """Security configuration."""
    
    secret_key: str = Field(default="dev-secret-key-change-in-production", description="Secret key for JWT")
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expire_minutes: int = Field(default=1440, description="JWT expiration time in minutes")
    
    # CORS
    allowed_origins: List[str] = Field(default=["*"], description="Allowed CORS origins")
    allowed_methods: List[str] = Field(default=["*"], description="Allowed HTTP methods")
    allowed_headers: List[str] = Field(default=["*"], description="Allowed headers")
    
    class Config:
        env_prefix = "SECURITY_"


class Settings(BaseSettings):
    """Main application settings."""
    
    # Application info
    app_name: str = Field(default="OmniBio Biomarker Discovery", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    environment: str = Field(default="development", description="Environment (development, staging, production)")
    
    # API configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_reload: bool = Field(default=True, description="API auto-reload in development")
    
    # Nested configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log format")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        @classmethod
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings,
        ):
            return (
                init_settings,
                env_settings,
                yaml_config_settings_source,
                file_secret_settings,
            )


def yaml_config_settings_source(settings: BaseSettings) -> Dict[str, Any]:
    """
    Load configuration from YAML files.
    Looks for config.yaml in the current directory and config/{environment}.yaml
    """
    config_data = {}
    
    # Load main config.yaml
    config_file = Path("config.yaml")
    if config_file.exists():
        with open(config_file, "r") as f:
            config_data.update(yaml.safe_load(f) or {})
    
    # Load environment-specific config
    env = os.getenv("ENVIRONMENT", "development")
    env_config_file = Path(f"config/{env}.yaml")
    if env_config_file.exists():
        with open(env_config_file, "r") as f:
            env_config = yaml.safe_load(f) or {}
            # Merge environment config (takes precedence)
            config_data.update(env_config)
    
    return config_data


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# For backward compatibility and easy imports
settings = get_settings() 