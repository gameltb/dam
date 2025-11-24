from dam.core.systems import system
from dam.core.world import World
from dam.core.database import DatabaseManager
from dam_domarkx.commands import ForkSession, CreateSession
from dam_domarkx.models.domarkx import Session, Message
from dam.models.core.entity import Entity
from sqlalchemy.future import select


@system(on_command=CreateSession)
async def create_session(cmd: CreateSession, world: World, db: DatabaseManager) -> Entity:
    async with db.get_db_session() as session:
        new_session_entity = Entity()
        new_session = Session(
            workspace_id=cmd.workspace_id,
            parent_id=None,
        )
        new_session.entity = new_session_entity
        session.add(new_session)
        await session.commit()
        return new_session_entity


@system(on_command=ForkSession)
async def fork_session(cmd: ForkSession, world: World, db: DatabaseManager) -> Entity:
    async with db.get_db_session() as session:
        result = await session.execute(select(Session).where(Session.session_id == cmd.session_id))
        parent_session = result.scalars().first()
        if parent_session is None:
            raise ValueError(f"Session with id {cmd.session_id} not found.")

        new_session_entity = Entity()
        new_session = Session(
            workspace_id=parent_session.workspace_id,
            parent_id=parent_session.session_id,
        )
        new_session.entity = new_session_entity
        session.add(new_session)

        result = await session.execute(select(Message).where(Message.session_id == parent_session.session_id))
        for message in result.scalars().all():
            new_message = Message(
                session_id=new_session.session_id,
                role=message.role,
                content=message.content,
            )
            new_message.entity = Entity()
            session.add(new_message)

        await session.commit()
        return new_session_entity
