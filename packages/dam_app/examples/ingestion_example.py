import asyncio
import logging
from io import BytesIO
from pathlib import Path

from dam.core.config import AppSettings, WorldConfig
from dam.core.logging_config import setup_logging
from dam.core.world import World, clear_world_registry
from dam.services import ecs_service
from dam_fs import FsPlugin
from dam_media_image import ImagePlugin
from dam_media_image.models.properties.image_dimensions_component import ImageDimensionsComponent
from PIL import Image
from sqlalchemy import select

from dam_app import AppPlugin
from dam_app.commands import IngestAssetStreamCommand


async def main():
    """
    An example demonstrating the new event-driven ingestion pipeline.
    """
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # --- 1. World Setup ---
    # In a real app, this would come from a config file.
    # We create a temporary in-memory database for this example.
    clear_world_registry()
    temp_dir = Path("./temp_dam_example")
    world_config = WorldConfig(
        name="example_world",
        db_url=f"sqlite+aiosqlite:///{temp_dir}/dam.db",
        ASSET_STORAGE_PATH=str(temp_dir / "assets"),
    )
    app_settings = AppSettings(worlds=[world_config], DEFAULT_WORLD_NAME="example_world", TESTING_MODE=True)

    world = World(world_config, app_settings)

    # Register all the plugins needed for the pipeline
    world.add_plugin(FsPlugin())
    world.add_plugin(ImagePlugin())
    world.add_plugin(AppPlugin())  # AppPlugin registers the ingestion systems

    # Create the database schema
    await world.create_db_and_tables()
    logger.info(f"World '{world.name}' created and plugins registered.")

    # --- 2. Create a dummy asset ---
    # Create a simple 10x10 black PNG image in memory
    img = Image.new("RGB", (10, 10), color="black")
    in_memory_stream = BytesIO()
    img.save(in_memory_stream, format="PNG")
    in_memory_stream.seek(0)
    logger.info("Created a dummy 10x10 PNG image in memory.")

    # --- 3. Dispatch the ingestion event ---
    async with world.db_session_maker() as session:
        # Create an entity for our new asset
        new_entity = await ecs_service.create_entity(session)
        await session.flush()
        logger.info(f"Created new entity with ID: {new_entity.id}")

        # Create the initial command
        ingestion_command = IngestAssetStreamCommand(
            entity=new_entity,
            file_content=in_memory_stream,
            original_filename="dummy_image.png",
            world_name=world.name,
        )

        # Dispatch the command to the world
        # The scheduler will pick it up and trigger the command handler
        await world.dispatch_command(ingestion_command)
        logger.info("Dispatched IngestAssetStreamCommand.")

        # In a real long-running app, you wouldn't commit here.
        # But for this script, we commit to save the results of the event processing.
        await session.commit()
        logger.info("Session committed.")

    # --- 4. Verify the result ---
    # Check if the ImageDimensionsComponent was added correctly by the pipeline
    async with world.db_session_maker() as session:
        logger.info("Verifying the result...")
        stmt = select(ImageDimensionsComponent).where(ImageDimensionsComponent.entity_id == new_entity.id)
        result = await session.execute(stmt)
        image_dims_comp = result.scalars().one_or_none()

        if image_dims_comp:
            logger.info("✅ Verification successful!")
            logger.info(
                f"Found ImageDimensionsComponent: width={image_dims_comp.width}, height={image_dims_comp.height}"
            )
            assert image_dims_comp.width == 10
            assert image_dims_comp.height == 10
        else:
            logger.error("❌ Verification failed! ImageDimensionsComponent not found.")

    # --- Cleanup ---
    import shutil

    shutil.rmtree(temp_dir)
    logger.info("Cleaned up temporary directory.")


if __name__ == "__main__":
    asyncio.run(main())
