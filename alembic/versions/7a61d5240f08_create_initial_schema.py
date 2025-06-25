"""create_initial_schema

Revision ID: 0001_create_initial_schema
Revises:
Create Date: 2025-06-25 10:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# from dam.models.types import TZDateTime  # This was incorrect; use sa.DateTime(timezone=True)

# revision identifiers, used by Alembic.
revision: str = "7a61d5240f08"  # Updated to match filename
down_revision: Union[str, Sequence[str], None] = None  # This is the first revision
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create all tables from current models."""
    # entities table
    op.create_table(
        "entities",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Common columns for components inheriting from BaseComponent
    # id, entity_id, created_at, updated_at

    # content_hashes table (for ContentHashComponent)
    op.create_table(
        "content_hashes",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("entity_id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
        sa.Column("hash_type", sa.String(length=64), nullable=False),
        sa.Column("hash_value", sa.String(length=256), nullable=False),
        sa.ForeignKeyConstraint(
            ["entity_id"],
            ["entities.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("hash_type", "hash_value", name="uq_content_hash_type_value"),
        sa.Index("ix_content_hashes_entity_id", ["entity_id"], unique=False),
        sa.Index("ix_content_hashes_hash_value", ["hash_value"], unique=False),
    )

    # image_perceptual_hashes table (for ImagePerceptualHashComponent)
    op.create_table(
        "image_perceptual_hashes",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("entity_id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
        sa.Column("hash_type", sa.String(length=64), nullable=False),  # e.g., "phash", "ahash", "dhash"
        sa.Column("hash_value", sa.String(length=256), nullable=False),  # Hash value as string
        sa.ForeignKeyConstraint(
            ["entity_id"],
            ["entities.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("entity_id", "hash_type", name="uq_image_phash_entity_type"),
        sa.Index("ix_image_perceptual_hashes_entity_id", ["entity_id"], unique=False),
    )

    # file_locations table (for FileLocationComponent) - FINAL SCHEMA
    op.create_table(
        "file_locations",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("entity_id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
        sa.Column("file_identifier", sa.String(length=256), nullable=False),  # SHA256 hash
        sa.Column(
            "storage_type",
            sa.String(length=64),
            default="local_content_addressable",
            server_default="local_content_addressable",
            nullable=False,
        ),
        sa.Column("original_filename", sa.String(length=1024), nullable=True),
        sa.ForeignKeyConstraint(
            ["entity_id"],
            ["entities.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("entity_id", "file_identifier", name="uq_file_location_entity_identifier"),
        sa.Index("ix_file_locations_entity_id", ["entity_id"], unique=False),
    )

    # file_properties table (for FilePropertiesComponent)
    op.create_table(
        "file_properties",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("entity_id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
        sa.Column("original_filename", sa.String(length=1024), nullable=True),
        sa.Column("file_size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("mime_type", sa.String(length=255), nullable=True),
        # Assuming one FileProperties per entity; if multiple, remove unique constraint on entity_id
        sa.ForeignKeyConstraint(
            ["entity_id"],
            ["entities.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("entity_id", name="uq_file_properties_entity"),
        sa.Index("ix_file_properties_entity_id", ["entity_id"], unique=True),  # if one-to-one
    )


def downgrade() -> None:
    """Drop all tables."""
    op.drop_table("file_properties")
    op.drop_table("file_locations")
    op.drop_table("image_perceptual_hashes")
    op.drop_table("content_hashes")
    op.drop_table("entities")
