import hashlib
import json
from dam.core.database import DatabaseManager
from dam_domarkx.models.domarkx import Workspace, Resource
from sqlalchemy.future import select


async def hash_workspace(workspace: Workspace, db: DatabaseManager) -> str:
    """
    Hashes the workspace and its resources to generate a stable, content-based hash.
    """
    hasher = hashlib.sha1()
    hasher.update(workspace.name.encode())

    async with db.get_db_session() as session:
        result = await session.execute(select(Resource).where(Resource.workspace_id == workspace.workspace_id))
        resources = result.scalars().all()
        for resource in sorted(resources, key=lambda r: r.resource_id):
            hasher.update(str(resource.resource_id).encode())
            hasher.update(json.dumps(resource.config, sort_keys=True).encode())

    return hasher.hexdigest()
