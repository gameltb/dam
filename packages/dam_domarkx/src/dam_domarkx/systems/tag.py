from dam.core.systems import system
from dam.core.world import World
from dam.core.database import DatabaseManager
from dam_domarkx.commands import CreateTag, GetTags
from dam_domarkx.models.git import Tag
from dam.models.core.entity import Entity
from sqlalchemy.future import select
from typing import List


@system(on_command=CreateTag)
async def create_tag(cmd: CreateTag, world: World, db: DatabaseManager) -> Entity:
    async with db.get_db_session() as session:
        new_tag_entity = Entity()
        new_tag = Tag(
            workspace_id=cmd.workspace_id,
            name=cmd.name,
            commit_id=cmd.commit_id,
        )
        new_tag.entity = new_tag_entity
        session.add(new_tag)
        await session.commit()
        return new_tag_entity


@system(on_command=GetTags)
async def get_tags(cmd: GetTags, world: World, db: DatabaseManager) -> List[Tag]:
    async with db.get_db_session() as session:
        result = await session.execute(select(Tag).where(Tag.workspace_id == cmd.workspace_id))
        return result.scalars().all()
