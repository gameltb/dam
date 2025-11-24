from dam.core.plugin import Plugin
from dam.core.world import World
from dam_domarkx.systems.versioning import WorkspaceVersioningSystem
from dam_domarkx.systems.create_workspace import create_workspace
from dam_domarkx.systems.session import fork_session, create_session
from dam_domarkx.systems.tag import create_tag, get_tags
from dam_domarkx.commands import CreateWorkspace, ForkSession, CreateSession, CreateTag, GetTags


class DomarkxPlugin(Plugin):
    def build(self, world: World):
        world.register_system(WorkspaceVersioningSystem)
        world.register_system(create_workspace, command_type=CreateWorkspace)
        world.register_system(fork_session, command_type=ForkSession)
        world.register_system(create_session, command_type=CreateSession)
        world.register_system(create_tag, command_type=CreateTag)
        world.register_system(get_tags, command_type=GetTags)
