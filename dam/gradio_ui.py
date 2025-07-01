import gradio as gr
from typing import Optional, List, Dict, Any
from dam.core.world import World
from dam.core.config import settings as app_settings
from dam.services import ecs_service, world_service, character_service, semantic_service, transcode_service
from dam.models.core.entity import Entity

# Helper function to get world choices
def get_world_choices() -> List[str]:
    return [world.name for world in app_settings.worlds] if app_settings.worlds else ["No worlds configured"]

# Helper function to get current world (placeholder)
def get_current_world(world_name: Optional[str]) -> Optional[World]:
    if not world_name or world_name == "No worlds configured":
        return None
    # This is a simplified way; actual world loading might be more complex
    # from dam.core.world_setup import get_world # Assuming get_world is accessible
    # return get_world(world_name)
    # For now, let's find it in app_settings.worlds
    if app_settings.worlds:
        for world_instance in app_settings.worlds:
            if world_instance.name == world_name:
                return world_instance
    return None

# Global variable to store the current world (alternative to passing it around in Gradio state)
# This might need refinement for multi-user scenarios or if Gradio state is preferred
_active_world: Optional[World] = None

def set_active_world(world_name: str) -> str:
    global _active_world
    _active_world = get_current_world(world_name)
    if _active_world:
        return f"World '{_active_world.name}' selected."
    return "Failed to select world or no world chosen."

# --- Asset Listing and Filtering ---
async def list_assets(world_name: Optional[str], filename_filter: str, mime_type_filter: str, current_page: int = 1, page_size: int = 20) -> gr.DataFrame:
    # world = get_current_world(world_name) # Using global _active_world now
    if not _active_world:
        return gr.DataFrame(value=None, headers=["ID", "Filename", "MIME Type"], label="Assets")

    # This is a simplified example. Real implementation would involve async DB calls.
    # assets_data = []
    # async with _active_world.get_db_session() as session:
    #     # Simplified query; actual query would be more complex with filters and pagination
    #     from dam.models.properties.file_properties_component import FilePropertiesComponent
    #     from sqlalchemy import select
    #     query = select(FilePropertiesComponent.entity_id, FilePropertiesComponent.original_filename, FilePropertiesComponent.mime_type)
    #     if filename_filter:
    #         query = query.filter(FilePropertiesComponent.original_filename.ilike(f"%{filename_filter}%"))
    #     if mime_type_filter:
    #         query = query.filter(FilePropertiesComponent.mime_type == mime_type_filter)

    #     # Add pagination logic here
    #     # query = query.offset((current_page - 1) * page_size).limit(page_size)

    #     result = await session.execute(query)
    #     all_rows = result.all()
    #     assets_data = [(row.entity_id, row.original_filename, row.mime_type) for row in all_rows]

    # Placeholder data for now
    if _active_world.name == "default":
        assets_data = [
            (1, "image.jpg", "image/jpeg"),
            (2, "document.pdf", "application/pdf"),
            (3, "video.mp4", "video/mp4"),
        ]
        if filename_filter:
            assets_data = [ad for ad in assets_data if filename_filter.lower() in ad[1].lower()]
        if mime_type_filter:
            assets_data = [ad for ad in assets_data if mime_type_filter == ad[2]]

    else:
        assets_data = []


    df_value = {"data": assets_data, "headers": ["ID", "Filename", "MIME Type"]}
    return gr.DataFrame(value=df_value) # , label=f"Assets (Page {current_page})")

async def get_asset_details(world_name: Optional[str], asset_id: Optional[int]) -> gr.JSON:
    # world = get_current_world(world_name) # Using global _active_world now
    if not _active_world or asset_id is None:
        return gr.JSON(value=None, label="Asset Components")

    components_data = {}
    # async with _active_world.get_db_session() as session:
    #     from dam.models.core.base_component import REGISTERED_COMPONENT_TYPES
    #     entity = await ecs_service.get_entity(session, asset_id)
    #     if entity:
    #         for comp_type_cls in REGISTERED_COMPONENT_TYPES:
    #             comp_type_name = comp_type_cls.__name__
    #             components = await ecs_service.get_components(session, asset_id, comp_type_cls)
    #             component_instances_data = []
    #             if components:
    #                 for comp_instance in components:
    #                     instance_data = {c.key: getattr(comp_instance, c.key) for c in comp_instance.__table__.columns if not c.key.startswith("_")}
    #                     component_instances_data.append(instance_data)
    #             components_data[comp_type_name] = component_instances_data

    # Placeholder
    if _active_world.name == "default" and asset_id == 1:
        components_data = {
            "FilePropertiesComponent": [{"entity_id": 1, "original_filename": "image.jpg", "mime_type": "image/jpeg", "size_bytes": 102400}],
            "ImageDimensionsComponent": [{"entity_id": 1, "width": 1920, "height": 1080}]
        }
    elif _active_world.name == "default" and asset_id == 2:
         components_data = {
            "FilePropertiesComponent": [{"entity_id": 2, "original_filename": "document.pdf", "mime_type": "application/pdf", "size_bytes": 204800}],
        }

    return gr.JSON(value=components_data, label=f"Components for Asset ID: {asset_id}")

# --- Dialog Functions (Placeholders) ---
def add_assets_ui(world_name: Optional[str], files: List[gr.File]) -> str:
    # world = get_current_world(world_name)
    if not _active_world: return "No world selected."
    if not files: return "No files selected."
    # In a real app, process files: ecs_service.add_asset_from_file_paths
    return f"Simulated adding {len(files)} assets to '{_active_world.name}'. Paths: {[f.name for f in files]}"

def find_by_hash_ui(world_name: Optional[str], hash_value: str, hash_type: str) -> str:
    # world = get_current_world(world_name)
    if not _active_world: return "No world selected."
    # Real: search by hash
    return f"Simulated searching for hash '{hash_value}' (type: {hash_type}) in '{_active_world.name}'."

def find_similar_images_ui(world_name: Optional[str], image_file: gr.Image) -> str:
    # world = get_current_world(world_name)
    if not _active_world: return "No world selected."
    if not image_file: return "No image uploaded for similarity search."
    # Real: find similar images
    return f"Simulated finding similar images to the uploaded image in '{_active_world.name}'."

def transcode_asset_ui(world_name: Optional[str], asset_id: int, profile_name: str) -> str:
    # world = get_current_world(world_name)
    if not _active_world: return "No world selected."
    # Real: transcode_service.transcode_asset
    return f"Simulated transcoding asset ID {asset_id} using profile '{profile_name}' in '{_active_world.name}'."

def manage_characters_ui(world_name: Optional[str], action: str, character_name: Optional[str]=None, asset_id: Optional[int]=None) -> str:
    # world = get_current_world(world_name)
    if not _active_world: return "No world selected."
    # Real: character_service actions
    return f"Simulated character management: Action '{action}', Character '{character_name}', Asset ID '{asset_id}' in '{_active_world.name}'."

def semantic_search_ui(world_name: Optional[str], query: str) -> str:
    # world = get_current_world(world_name)
    if not _active_world: return "No world selected."
    # Real: semantic_service.search_by_text
    return f"Simulated semantic search for '{query}' in '{_active_world.name}'."

def evaluation_setup_ui(world_name: Optional[str], params: str) -> str: # Params as JSON string for simplicity
    # world = get_current_world(world_name)
    if not _active_world: return "No world selected."
    # Real: setup evaluation
    return f"Simulated evaluation setup with params '{params}' in '{_active_world.name}'."

# --- World Operations (Placeholders) ---
def export_world_ui(world_name: Optional[str], export_path: str) -> str:
    # world = get_current_world(world_name)
    if not _active_world: return "No world selected."
    # Real: world_service.export_world_data
    return f"Simulated exporting world '{_active_world.name}' to '{export_path}'."

def import_world_ui(world_name: Optional[str], import_file: gr.File) -> str:
    # world = get_current_world(world_name)
    if not _active_world: return "No world selected."
    if not import_file: return "No import file provided."
    # Real: world_service.import_world_data
    return f"Simulated importing data from '{import_file.name}' into world '{_active_world.name}'."

def merge_worlds_ui(target_world_name: str, source_world_name: str, delete_source: bool) -> str:
    # target_world = get_current_world(target_world_name)
    # source_world = get_current_world(source_world_name)
    # For now, assume _active_world is the target
    if not _active_world: return "Target world not selected."
    source_world = get_current_world(source_world_name)
    if not source_world: return f"Source world '{source_world_name}' not found."
    if _active_world.name == source_world.name: return "Target and source worlds cannot be the same."
    # Real: world_service.merge_worlds
    return f"Simulated merging world '{source_world.name}' into '{_active_world.name}'. Delete source: {delete_source}."

def split_world_ui(source_world_name: str, criteria: str, new_world_name_1: str, new_world_name_2: str) -> str:
    # source_world = get_current_world(source_world_name) # _active_world is source
    if not _active_world: return "Source world not selected."
    # Real: world_service.split_world
    return f"Simulated splitting world '{_active_world.name}' based on '{criteria}' into '{new_world_name_1}' and '{new_world_name_2}'."

def setup_db_ui(world_name: Optional[str]) -> str:
    # world = get_current_world(world_name)
    if not _active_world: return "No world selected."
    # Real: _active_world.create_db_and_tables()
    return f"Simulated database setup for world '{_active_world.name}'."

# --- Gradio Interface Definition ---
def create_dam_ui():
    with gr.Blocks(title="ECS DAM System") as dam_interface:
        gr.Markdown("# ECS Digital Asset Management System")

        with gr.Row():
            world_selector = gr.Dropdown(choices=get_world_choices(), label="Select DAM World", value=None if not get_world_choices() or get_world_choices()[0] == "No worlds configured" else get_world_choices()[0])
            world_status_output = gr.Textbox(label="World Status", interactive=False)

        # Connect world selector to set the active world
        # And also trigger initial asset list load for the selected world
        # world_selector.change(set_active_world, inputs=world_selector, outputs=world_status_output).then(
        #     list_assets, inputs=[world_selector, gr.Textbox(value=""), gr.Textbox(value="")], outputs=assets_df # How to pass initial filters?
        # )
        # A bit complex to chain directly like this with initial values.
        # Let's use a button for explicit world selection confirmation for now.
        confirm_world_button = gr.Button("Confirm World Selection")


        with gr.Tabs() as main_tabs:
            with gr.TabItem("Assets"):
                with gr.Row():
                    filename_filter_input = gr.Textbox(label="Filter by Filename")
                    mime_type_filter_input = gr.Dropdown(label="Filter by MIME Type", choices=["", "image/jpeg", "application/pdf", "video/mp4"]) # TODO: Populate dynamically
                    # page_number_input = gr.Number(label="Page", value=1, minimum=1, step=1, interactive=True)
                    # page_size_input = gr.Number(label="Assets per Page", value=20, minimum=5, maximum=100, step=5, interactive=True)

                load_assets_button = gr.Button("Load/Refresh Assets")

                assets_df = gr.DataFrame(headers=["ID", "Filename", "MIME Type"], label="Assets", interactive=False, row_count=(20, "dynamic"))
                asset_detail_json = gr.JSON(label="Asset Components")

                # Connecting asset loading
                # load_assets_button.click(list_assets, inputs=[world_selector, filename_filter_input, mime_type_filter_input, page_number_input, page_size_input], outputs=assets_df)
                load_assets_button.click(list_assets, inputs=[world_selector, filename_filter_input, mime_type_filter_input], outputs=assets_df)


                # When a row is selected in the DataFrame, show its details
                # This requires interactive=True on DataFrame, and then using .select event
                # For now, let's use a separate input for Asset ID to view details
                with gr.Row():
                    asset_id_for_detail = gr.Number(label="Enter Asset ID to View Details", minimum=1, step=1)
                    view_detail_button = gr.Button("View Details")

                view_detail_button.click(get_asset_details, inputs=[world_selector, asset_id_for_detail], outputs=asset_detail_json)


            with gr.TabItem("Operations"):
                with gr.Accordion("Add Assets", open=False):
                    # add_asset_files_input = gr.Files(label="Select Files to Add") # Gradio Files component
                    add_asset_files_input = gr.File(label="Select Files to Add", file_count="multiple")

                    add_asset_button = gr.Button("Add Selected Assets")
                    add_asset_output = gr.Textbox(label="Add Asset Status")
                    add_asset_button.click(add_assets_ui, inputs=[world_selector, add_asset_files_input], outputs=add_asset_output)

                with gr.Accordion("Find Asset by Hash", open=False):
                    hash_value_input = gr.Textbox(label="Hash Value")
                    hash_type_dropdown = gr.Dropdown(choices=["MD5", "SHA256", "aHash", "pHash", "dHash"], label="Hash Type", value="SHA256")
                    find_hash_button = gr.Button("Find by Hash")
                    find_hash_output = gr.Textbox(label="Find by Hash Result")
                    find_hash_button.click(find_by_hash_ui, inputs=[world_selector, hash_value_input, hash_type_dropdown], outputs=find_hash_output)

                with gr.Accordion("Find Similar Images", open=False):
                    similar_image_upload = gr.Image(type="filepath", label="Upload Image for Similarity Search")
                    find_similar_button = gr.Button("Find Similar")
                    find_similar_output = gr.Textbox(label="Similarity Search Result")
                    find_similar_button.click(find_similar_images_ui, inputs=[world_selector, similar_image_upload], outputs=find_similar_output)

                with gr.Accordion("Transcode Asset", open=False):
                    transcode_asset_id_input = gr.Number(label="Asset ID to Transcode", minimum=1, step=1)
                    # TODO: Transcode profiles should be dynamically loaded
                    transcode_profile_input = gr.Dropdown(label="Transcode Profile", choices=["default_avif", "default_jpegxl", "thumbnail_jpeg"])
                    transcode_button = gr.Button("Transcode Asset")
                    transcode_output = gr.Textbox(label="Transcode Status")
                    transcode_button.click(transcode_asset_ui, inputs=[world_selector, transcode_asset_id_input, transcode_profile_input], outputs=transcode_output)

                with gr.Accordion("Character Management", open=False):
                    # This would be more complex in reality, e.g., separate UIs for create, link, list
                    char_action_dropdown = gr.Dropdown(choices=["create", "link_to_asset", "unlink_from_asset", "list_for_asset", "list_all"], label="Action")
                    char_name_input = gr.Textbox(label="Character Name (for create, link, unlink)")
                    char_asset_id_input = gr.Number(label="Asset ID (for link, unlink, list_for_asset)", minimum=1, step=1)
                    char_manage_button = gr.Button("Execute Character Action")
                    char_manage_output = gr.Textbox(label="Character Management Result")
                    char_manage_button.click(manage_characters_ui, inputs=[world_selector, char_action_dropdown, char_name_input, char_asset_id_input], outputs=char_manage_output)

                with gr.Accordion("Semantic Search", open=False):
                    semantic_query_input = gr.Textbox(label="Semantic Search Query")
                    semantic_search_button = gr.Button("Search Semantically")
                    semantic_search_output = gr.Textbox(label="Semantic Search Results")
                    semantic_search_button.click(semantic_search_ui, inputs=[world_selector, semantic_query_input], outputs=semantic_search_output)

                with gr.Accordion("Transcoding Evaluation Setup", open=False):
                    # Simplified: parameters as a JSON string. Real UI would have structured inputs.
                    eval_params_input = gr.Textbox(label="Evaluation Parameters (JSON format)", lines=3, placeholder='{"profile_names": ["profile1", "profile2"], "metric_types": ["PSNR", "SSIM"]}')
                    eval_setup_button = gr.Button("Setup Evaluation")
                    eval_setup_output = gr.Textbox(label="Evaluation Setup Status")
                    eval_setup_button.click(evaluation_setup_ui, inputs=[world_selector, eval_params_input], outputs=eval_setup_output)


            with gr.TabItem("World Management"):
                with gr.Accordion("Export Current World", open=False):
                    export_path_input = gr.Textbox(label="Export File Path (e.g., /path/to/export.json)")
                    export_world_button = gr.Button("Export World")
                    export_world_output = gr.Textbox(label="Export Status")
                    export_world_button.click(export_world_ui, inputs=[world_selector, export_path_input], outputs=export_world_output)

                with gr.Accordion("Import Data into Current World", open=False):
                    import_file_input = gr.File(label="Select Import File (.json)")
                    import_world_button = gr.Button("Import Data")
                    import_world_output = gr.Textbox(label="Import Status")
                    import_world_button.click(import_world_ui, inputs=[world_selector, import_file_input], outputs=import_world_output)

                with gr.Accordion("Merge Worlds", open=False):
                    # Target world is the currently selected world (world_selector)
                    # TODO: Populate source world choices excluding the target world
                    all_worlds = get_world_choices()
                    source_world_merge_input = gr.Dropdown(choices=all_worlds, label="Source World to Merge From")
                    delete_source_checkbox = gr.Checkbox(label="Delete Source World After Merge", value=False)
                    merge_worlds_button = gr.Button("Merge Worlds")
                    merge_worlds_output = gr.Textbox(label="Merge Status")
                    merge_worlds_button.click(merge_worlds_ui, inputs=[world_selector, source_world_merge_input, delete_source_checkbox], outputs=merge_worlds_output)

                with gr.Accordion("Split Current World", open=False):
                    # Source world is the currently selected world (world_selector)
                    # Simplified: criteria as text, new world names
                    split_criteria_input = gr.Textbox(label="Split Criteria (e.g., tag:nature)")
                    split_new_world1_input = gr.Textbox(label="Name for New World 1 (e.g., nature_assets)")
                    split_new_world2_input = gr.Textbox(label="Name for New World 2 (e.g., other_assets)")
                    split_world_button = gr.Button("Split World")
                    split_world_output = gr.Textbox(label="Split Status")
                    split_world_button.click(split_world_ui, inputs=[world_selector, split_criteria_input, split_new_world1_input, split_new_world2_input], outputs=split_world_output)

                with gr.Accordion("Setup Database for Current World", open=False):
                    setup_db_button = gr.Button("Initialize/Setup Database")
                    setup_db_output = gr.Textbox(label="Database Setup Status")
                    setup_db_button.click(setup_db_ui, inputs=[world_selector], outputs=setup_db_output)

        # Logic for confirming world selection and initial load
        confirm_world_button.click(
            set_active_world, inputs=world_selector, outputs=world_status_output
        ).then(
            lambda ws: list_assets(ws, "", ""), # Call list_assets with current world_selector value and empty filters
            inputs=[world_selector],
            outputs=assets_df
        )


    return dam_interface

if __name__ == "__main__":
    # This part is for standalone testing of the Gradio UI.
    # It would need to initialize worlds similar to how the CLI does.
    print("Attempting to initialize worlds for standalone Gradio UI...")
    try:
        from dam.core import config as app_config
        from dam.core.world import get_world, create_and_register_all_worlds_from_settings
        from dam.core.world_setup import register_core_systems, get_world_by_name

        # Ensure worlds are initialized from settings
        initialized_worlds = create_and_register_all_worlds_from_settings(app_settings=app_config.settings)
        for world_instance in initialized_worlds:
            register_core_systems(world_instance) # Register systems for this world
            # For Gradio, we might need to ensure DB is set up if it's a new world
            # This is a simplified assumption; real setup might be manual via UI/CLI
            # asyncio.run(world_instance.create_db_and_tables())


        # Set a default active world if available for testing
        if app_config.settings.DEFAULT_WORLD_NAME:
            _active_world = get_world_by_name(app_config.settings.DEFAULT_WORLD_NAME)
            print(f"Standalone: Default world '{_active_world.name if _active_world else 'None'}' pre-selected.")
        elif initialized_worlds:
            _active_world = initialized_worlds[0]
            print(f"Standalone: First initialized world '{_active_world.name if _active_world else 'None'}' pre-selected.")
        else:
            print("Standalone: No worlds configured or could be initialized.")
            # Create a dummy 'default' world for UI testing if no config exists
            if not any(w.name == "default" for w in app_settings.worlds):
                 from dam.core.world import World
                 dummy_world = World(name="default", db_url="sqlite+aiosqlite:///:memory:")
                 app_settings.worlds.append(dummy_world) # This is a hack for standalone
                 _active_world = dummy_world
                 print("Standalone: Created a dummy 'default' world for UI testing.")


    except Exception as e:
        print(f"Standalone: Error initializing worlds: {e}")
        print("Ensure your .env or config file is set up correctly if you expect worlds to load.")
        # Create a dummy 'default' world if initialization failed, so UI can still launch
        if not _active_world and (not app_settings.worlds or not any(w.name == "default" for w in app_settings.worlds)):
            from dam.core.world import World
            _active_world = World(name="default", db_url="sqlite+aiosqlite:///:memory:") # In-memory for dummy
            app_settings.worlds.append(_active_world) # Hack for standalone
            print("Standalone: Created a fallback dummy 'default' world due to init errors.")


    # Setup logging for standalone UI run
    import logging
    from dam.core.logging_config import setup_logging
    setup_logging(level=logging.INFO)

    ui = create_dam_ui()
    ui.launch()

# To be called from CLI:
def launch_ui(world_name: Optional[str] = None):
    """
    Launches the Gradio UI.
    If world_name is provided, it attempts to pre-select that world.
    """
    global _active_world
    if world_name:
        from dam.core.world_setup import get_world_by_name # Ensure it's imported here for CLI use
        _active_world = get_world_by_name(world_name)
    elif app_settings.worlds: # If no specific world, pick first one if available
        _active_world = app_settings.worlds[0]


    interface = create_dam_ui()
    interface.launch()
