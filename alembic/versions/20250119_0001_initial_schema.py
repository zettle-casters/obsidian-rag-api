from __future__ import annotations

from alembic import op

revision = "20250119_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id VARCHAR(36) PRIMARY KEY,
            google_sub VARCHAR(255) UNIQUE,
            email VARCHAR(320) UNIQUE,
            name VARCHAR(255),
            avatar_url TEXT,
            is_demo BOOLEAN NOT NULL DEFAULT false,
            is_admin BOOLEAN NOT NULL DEFAULT false,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
    )
    op.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS is_admin BOOLEAN NOT NULL DEFAULT false;")

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id VARCHAR(36) PRIMARY KEY,
            user_id VARCHAR(36) REFERENCES users(id),
            token VARCHAR(128) UNIQUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            expires_at TIMESTAMPTZ NOT NULL
        );
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_sessions_user_id ON sessions (user_id);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_sessions_token ON sessions (token);")

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS oauth_states (
            id VARCHAR(36) PRIMARY KEY,
            state VARCHAR(128) UNIQUE,
            code_verifier VARCHAR(256),
            redirect_uri TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_oauth_states_state ON oauth_states (state);")

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS vaults (
            id VARCHAR(36) PRIMARY KEY,
            user_id VARCHAR(36) REFERENCES users(id),
            name VARCHAR(255),
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            include_paths JSONB NOT NULL DEFAULT '[]'::jsonb,
            exclude_paths JSONB NOT NULL DEFAULT '[]'::jsonb,
            chunk_size INTEGER NOT NULL DEFAULT 500,
            notes_count INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_vaults_user_id ON vaults (user_id);")

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS vault_files (
            id VARCHAR(36) PRIMARY KEY,
            vault_id VARCHAR(36) REFERENCES vaults(id),
            path TEXT,
            content_hash VARCHAR(64),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT uq_vault_files UNIQUE (vault_id, path)
        );
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_vault_files_vault_id ON vault_files (vault_id);")

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS mcp_tokens (
            id VARCHAR(36) PRIMARY KEY,
            user_id VARCHAR(36) REFERENCES users(id),
            token VARCHAR(128) UNIQUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_used_at TIMESTAMPTZ,
            CONSTRAINT uq_mcp_tokens_user_id UNIQUE (user_id)
        );
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_mcp_tokens_user_id ON mcp_tokens (user_id);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_mcp_tokens_token ON mcp_tokens (token);")

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS chats (
            id VARCHAR(36) PRIMARY KEY,
            user_id VARCHAR(36) REFERENCES users(id),
            vault_id VARCHAR(36) REFERENCES vaults(id),
            title VARCHAR(255) NOT NULL DEFAULT 'Новый чат',
            model_name VARCHAR(255),
            thread_id VARCHAR(64),
            is_shared BOOLEAN NOT NULL DEFAULT false,
            share_token VARCHAR(128) UNIQUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_chats_user_id ON chats (user_id);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_chats_vault_id ON chats (vault_id);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_chats_share_token ON chats (share_token);")

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_messages (
            id VARCHAR(36) PRIMARY KEY,
            chat_id VARCHAR(36) REFERENCES chats(id),
            role VARCHAR(16),
            content TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_chat_messages_chat_id ON chat_messages (chat_id);")

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_models (
            id VARCHAR(36) PRIMARY KEY,
            display_name VARCHAR(255) NOT NULL,
            system_name VARCHAR(255) NOT NULL UNIQUE,
            description TEXT,
            avatar_url TEXT,
            is_enabled BOOLEAN NOT NULL DEFAULT true,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS chat_messages;")
    op.execute("DROP TABLE IF EXISTS chats;")
    op.execute("DROP TABLE IF EXISTS llm_models;")
    op.execute("DROP TABLE IF EXISTS mcp_tokens;")
    op.execute("DROP TABLE IF EXISTS vault_files;")
    op.execute("DROP TABLE IF EXISTS vaults;")
    op.execute("DROP TABLE IF EXISTS oauth_states;")
    op.execute("DROP TABLE IF EXISTS sessions;")
    op.execute("DROP TABLE IF EXISTS users;")
