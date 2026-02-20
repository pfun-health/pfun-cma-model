"""empty message

Revision ID: 6a9f0bb4a9a5
Revises: 2bc0da65b9f2
Create Date: 2026-02-20 12:36:49.743096

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "6a9f0bb4a9a5"
down_revision: Union[str, Sequence[str], None] = "2bc0da65b9f2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "users",
        sa.Column("id", sa.INTEGER(), nullable=False),
        sa.Column("name", sa.VARCHAR(), nullable=False),
        sa.Column("email", sa.VARCHAR(), nullable=False),
        sa.Column("is_admin", sa.BOOLEAN(), nullable=False),
        sa.Column("site_id", sa.INTEGER(), nullable=True),
        sa.Column("age", sa.INTEGER(), nullable=False),
        sa.Column("bio", sa.VARCHAR(), nullable=True),
        sa.Column("hashed_password", sa.VARCHAR(), nullable=False),
        sa.ForeignKeyConstraint(
            ["site_id"],
            ["sites.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        if_not_exists=True,
    )
    op.create_index(
        op.f("ix_users_id"), "users", ["id"], unique=False, if_not_exists=True
    )
    op.create_index(
        op.f("ix_users_email"), "users", ["email"], unique=True, if_not_exists=True
    )
    op.create_table(
        "sites",
        sa.Column("id", sa.INTEGER(), nullable=False),
        sa.Column("name", sa.VARCHAR(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        if_not_exists=True,
    )
    op.create_index(
        op.f("ix_sites_id"), "sites", ["id"], unique=False, if_not_exists=True
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_sites_id"), table_name="sites")
    op.drop_table("sites")
    op.drop_index(op.f("ix_users_email"), table_name="users")
    op.drop_index(op.f("ix_users_id"), table_name="users")
    op.drop_table("users")
