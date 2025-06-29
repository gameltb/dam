"""rename_component_tables

Revision ID: b66de9bf8963
Revises:
Create Date: 2025-06-29 14:47:58.079125

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b66de9bf8963'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.rename_table('evaluation_results', 'component_evaluation_result')
    op.rename_table('evaluation_runs', 'component_evaluation_run')
    op.rename_table('transcode_profiles', 'component_transcode_profile')
    op.rename_table('transcoded_variants', 'component_transcoded_variant')
    op.rename_table('component_image_perceptual_hash_ahash', 'component_image_perceptual_ahash')
    op.rename_table('component_image_perceptual_hash_dhash', 'component_image_perceptual_dhash')
    op.rename_table('component_image_perceptual_hash_phash', 'component_image_perceptual_phash')
    op.rename_table('exiftool_metadata', 'component_exiftool_metadata')


def downgrade() -> None:
    """Downgrade schema."""
    op.rename_table('component_evaluation_result', 'evaluation_results')
    op.rename_table('component_evaluation_run', 'evaluation_runs')
    op.rename_table('component_transcode_profile', 'transcode_profiles')
    op.rename_table('component_transcoded_variant', 'transcoded_variants')
    op.rename_table('component_image_perceptual_ahash', 'component_image_perceptual_hash_ahash')
    op.rename_table('component_image_perceptual_dhash', 'component_image_perceptual_hash_dhash')
    op.rename_table('component_image_perceptual_phash', 'component_image_perceptual_hash_phash')
    op.rename_table('component_exiftool_metadata', 'exiftool_metadata')
