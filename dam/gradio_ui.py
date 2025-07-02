import gradio as gr
from typing import Optional, List, Dict, Any, Union # Added Union
import asyncio
from pathlib import Path
import uuid # For generating request_ids for queries


from dam.core.world import World
from dam.core.config import settings as app_settings
from dam.services import ecs_service, world_service, character_service, semantic_service, transcode_service
from dam.models.core.entity import Entity
from dam.core.events import AssetFileIngestionRequested, AssetReferenceIngestionRequested, FindEntityByHashQuery, FindSimilarImagesQuery, SemanticSearchQuery
from dam.services import file_operations
import dam.core.stages # Added import
from dam.services.transcode_service import TranscodeServiceError
from dam.services.character_service import CharacterConceptNotFoundError, CharacterLinkNotFoundError # Corrected import
from dam.services import ecs_service as dam_ecs_service # For EntityNotFoundError
from dam.models.properties import FilePropertiesComponent
from dam.systems import evaluation_systems
from dam.systems.evaluation_systems import EvaluationError

# Helper function to get world choices
def get_world_choices() -> List[str]:
    # app_settings.worlds is Dict[str, WorldConfig]
    return [wc.name for wc in app_settings.worlds.values()] if app_settings.worlds else ["Info: No worlds configured"]

# Helper function to get current world
def get_current_world(world_name: Optional[str]) -> Optional[World]:
    if not world_name or world_name.startswith("Info:") or world_name.startswith("Error:"):
        return None
    # Import get_world here to avoid circular dependency at module level if world.py imports gradio_ui indirectly
    from dam.core.world import get_world
    return get_world(world_name)

_active_world: Optional[World] = None

async def set_active_world_and_refresh_dropdowns(world_name: str) -> tuple[str, gr.Dropdown, gr.Dropdown, gr.Dropdown, gr.Dropdown]:
    global _active_world
    _active_world = get_current_world(world_name)

    # Initial states for dropdowns
    mime_type_choices = [""]
    transcode_profile_choices = ["Info: Select world first or refresh"]
    eval_run_choices = ["Info: Select world first or refresh"]

    status_message = f"Info: Processing world selection..."

    if _active_world:
        status_message = f"Success: World '{_active_world.name}' selected."
        try:
            async with _active_world.get_db_session() as session:
                from sqlalchemy import select
                from dam.models.properties.file_properties_component import FilePropertiesComponent
                stmt = select(FilePropertiesComponent.mime_type).distinct().order_by(FilePropertiesComponent.mime_type)
                result = await session.execute(stmt)
                # Assuming result.scalars() might be async or return an awaitable before .all()
                # It's safer to await it if the error indicates a coroutine.
                # However, typical SQLAlchemy 2.0 async has .scalars() sync after awaited execute.
                # Let's try with await on scalars() if the error persists.
                # For now, keeping as is, but if error is 'coroutine' object has no attribute 'all'
                # on result.scalars(), then await result.scalars() is needed.
                # The error was: 'coroutine' object has no attribute 'all' referring to the object before .all()
                # result.scalars() is synchronous after await session.execute()
                # and returns a result object on which .all() is also synchronous.
                scalar_results_obj = result.scalars() # No await
                all_items = scalar_results_obj.all()    # No await
                distinct_mime_types = []
                if all_items: # Check if it's not None or empty
                    for s in all_items:
                        if s:
                            distinct_mime_types.append(s)

                if distinct_mime_types:
                    mime_type_choices.extend(distinct_mime_types)
                else:
                     mime_type_choices = ["", "Info: No MIME types found"]
            status_message += " MIME types refreshed."
        except Exception as e:
            status_message += f" Error loading MIME types: {str(e)}"
            mime_type_choices = ["", f"Error: Could not load MIME types"]

        # Attempt to refresh other dropdowns
        transcode_profile_choices = await get_transcode_profile_choices()
        eval_run_choices = await get_evaluation_run_choices()

    else:
        status_message = "Error: Failed to select world or no valid world chosen."

    return (
        status_message,
        gr.Dropdown(choices=sorted(list(set(mime_type_choices))), value=""),
        gr.Dropdown(choices=transcode_profile_choices, value=transcode_profile_choices[0] if transcode_profile_choices else None),
        gr.Dropdown(choices=eval_run_choices, value=eval_run_choices[0] if eval_run_choices else None),
        gr.Dropdown(choices=eval_run_choices, value=eval_run_choices[0] if eval_run_choices else None)
    )

# --- Asset Listing and Filtering ---
async def list_assets_gr(filename_filter: str, mime_type_filter: str, current_page: int = 1, page_size: int = 20) -> tuple[gr.DataFrame, str]:
    if not _active_world:
        return gr.DataFrame(value=None, headers=["ID", "Filename", "MIME Type"]), "Error: No active world selected. Please select a world first."
    if current_page < 1:
        current_page = 1 # Ensure page is positive

    assets_data = []
    total_count = 0
    try:
        async with _active_world.get_db_session() as session:
            from dam.models.properties.file_properties_component import FilePropertiesComponent
            from sqlalchemy import select, func

            # Added FilePropertiesComponent.file_size_bytes and FilePropertiesComponent.created_at
            query = select(
                FilePropertiesComponent.entity_id,
                FilePropertiesComponent.original_filename,
                FilePropertiesComponent.mime_type,
                FilePropertiesComponent.file_size_bytes, # Corrected field name
                FilePropertiesComponent.created_at       # Corrected field name (inherited)
            )
            count_query = select(func.count(FilePropertiesComponent.entity_id))

            if filename_filter and filename_filter.strip():
                like_pattern = f"%{filename_filter.strip()}%"
                query = query.filter(FilePropertiesComponent.original_filename.ilike(like_pattern))
                count_query = count_query.filter(FilePropertiesComponent.original_filename.ilike(like_pattern))

            if mime_type_filter and mime_type_filter.strip() and not mime_type_filter.startswith("Info:") and not mime_type_filter.startswith("Error:"):
                query = query.filter(FilePropertiesComponent.mime_type == mime_type_filter)
                count_query = count_query.filter(FilePropertiesComponent.mime_type == mime_type_filter)

            total_count_result = await session.execute(count_query)
            total_count = total_count_result.scalar_one_or_none() or 0

            query = query.order_by(FilePropertiesComponent.original_filename) # Default sort
            query = query.offset((current_page - 1) * page_size).limit(page_size)

            result = await session.execute(query)
            # Fetching more details: size and creation date
            # Need to join with Entity table for creation date if it's there, or use FilePropertiesComponent.creation_date if available
            # For now, let's assume FilePropertiesComponent has `size_bytes` and `creation_date`
            # If `creation_date` is not directly on FilePropertiesComponent, this needs adjustment
            # Based on schema, FilePropertiesComponent has `creation_date` and `modification_date`

            assets_data = []
            for row_tuple in result.all(): # Assuming result.all() returns list of Row objects
                # Access by attribute name for clarity, ensure these attributes exist on the SELECTed component
                entity_id = row_tuple[0]
                original_filename = row_tuple[1]
                mime_type = row_tuple[2]
                file_size_bytes = row_tuple[3]  # Corrected variable name
                created_at = row_tuple[4]       # Corrected variable name

                # Format created_at if it's a datetime object
                formatted_created_at = created_at.strftime("%Y-%m-%d %H:%M:%S") if created_at else "N/A"

                assets_data.append((entity_id, original_filename, mime_type, file_size_bytes, formatted_created_at))


        status_message = f"Info: Displaying {len(assets_data)} of {total_count} assets (Page {current_page})."
        if not assets_data and total_count == 0 :
             status_message = "Info: No assets found matching criteria."
        elif not assets_data and total_count > 0 and current_page > 1 :
            status_message = f"Info: No assets on page {current_page} (Total: {total_count}). Try a lower page number."

        headers = ["ID", "Filename", "MIME Type", "File Size (Bytes)", "Created At"] # Updated header text
        df_value = {"data": assets_data, "headers": headers}

        # Enable sorting for Gradio DataFrame (Note: client-side sorting for the current page)
        # Gradio's DataFrame doesn't have a direct server-side sort toggle by clicking headers in the same way
        # some other dataframe libraries might. Sorting is typically handled by re-querying the data.
        # The `interactive=True` makes cells selectable, not headers sortable by default.
        # For now, the sorting is fixed by `order_by` in the query.
        # If more dynamic sorting is needed, the UI would need sort controls that re-trigger `list_assets_gr`.
        return gr.DataFrame(value=df_value, label=f"Assets (Page {current_page})", headers=headers), status_message
    except Exception as e:
        # Log the error
        print(f"Error in list_assets_gr: {str(e)}")
        import traceback
        traceback.print_exc()
        return gr.DataFrame(value=None, headers=["ID", "Filename", "MIME Type"]), f"Error: Could not load assets. Details: {str(e)}"

async def get_asset_details_gr(evt: gr.SelectData) -> gr.JSON:
    if not _active_world:
        return gr.JSON(value={"error": "Error: No active world selected."}, label="Asset Details")
    if evt.value is None or not isinstance(evt.index, tuple) or len(evt.index) != 2:
        return gr.JSON(value={"info": "Info: Click on an Asset ID in the table above to view its details."}, label="Asset Details")

    try:
        asset_id = int(evt.value) # Assuming evt.value is the ID from the first column
    except ValueError:
        return gr.JSON(value={"error": f"Error: Invalid Asset ID format from table selection: {evt.value}"}, label="Asset Details")

    tree_data = {}
    try:
        async with _active_world.get_db_session() as session:
            entity = await ecs_service.get_entity(session, asset_id)
            if not entity:
                return gr.JSON(value={"error": f"Error: Asset (Entity ID: {asset_id}) not found."}, label="Asset Details")

            entity_node_label = f"Entity ID: {entity.id}"
            tree_data[entity_node_label] = {}

            from dam.models.core.base_component import REGISTERED_COMPONENT_TYPES
            all_components_on_entity = await ecs_service.get_all_components_for_entity(session, asset_id)

            # Group components by type name
            components_by_type = {}
            for comp_instance in all_components_on_entity:
                comp_type_name = comp_instance.__class__.__name__
                if comp_type_name not in components_by_type:
                    components_by_type[comp_type_name] = []
                components_by_type[comp_type_name].append(comp_instance)

            if not components_by_type:
                tree_data[entity_node_label] = {"info": "No components found for this entity."}
                return gr.JSON(value=tree_data, label=f"Details for Entity ID: {asset_id}")

            for comp_type_name, component_instances in components_by_type.items():
                component_type_node_label = f"Component Type: {comp_type_name}"
                tree_data[entity_node_label][component_type_node_label] = {}

                for i, comp_instance in enumerate(component_instances):
                    # If multiple instances of the same component type, distinguish them.
                    # Usually, there's one, but the system might allow multiple for some types.
                    instance_label_suffix = f" (Instance {i+1})" if len(component_instances) > 1 else ""
                    component_instance_node_label = f"Instance Data{instance_label_suffix}"

                    # For components that have an 'id' or a 'name' attribute, use it for better labeling if possible.
                    # This is a heuristic.
                    if hasattr(comp_instance, 'id') and getattr(comp_instance, 'id') is not None:
                         component_instance_node_label = f"Instance (ID: {getattr(comp_instance, 'id')}){instance_label_suffix}"
                    elif hasattr(comp_instance, 'name') and getattr(comp_instance, 'name') is not None:
                         component_instance_node_label = f"Instance (Name: {getattr(comp_instance, 'name')}){instance_label_suffix}"
                    elif hasattr(comp_instance, 'profile_name') and getattr(comp_instance, 'profile_name') is not None: # Specific for TranscodeProfileComponent
                         component_instance_node_label = f"Instance (Profile: {getattr(comp_instance, 'profile_name')}){instance_label_suffix}"


                    tree_data[entity_node_label][component_type_node_label][component_instance_node_label] = {}
                    instance_data_node = tree_data[entity_node_label][component_type_node_label][component_instance_node_label]

                    attributes = {}
                    for c in comp_instance.__table__.columns:
                        if not c.key.startswith("_sa_"): # Exclude SQLAlchemy internal attributes
                            value = getattr(comp_instance, c.key)
                            attr_name = c.key
                            if isinstance(value, bytes):
                                try: attributes[attr_name] = value.decode('utf-8', errors='replace')
                                except: attributes[attr_name] = f"<bytes data of length {len(value)}>"
                            elif not isinstance(value, (str, int, float, bool, type(None), list, dict)):
                                attributes[attr_name] = str(value) # Convert other complex types to string
                            else:
                                attributes[attr_name] = value
                    instance_data_node["Attributes"] = attributes

        if not tree_data[entity_node_label]: # Should not happen if entity exists, but as a fallback
             tree_data[entity_node_label] = {"info": "No components found for this entity."}

        return gr.JSON(value=tree_data, label=f"Details for Entity ID: {asset_id}")
    except Exception as e:
        # Log the full error for debugging
        print(f"Error generating asset detail tree for ID {asset_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return gr.JSON(value={"error": f"Error: Could not fetch or structure details for Entity ID {asset_id}. Details: {str(e)}"}, label="Asset Details")

# --- Operation Functions ---
async def add_assets_ui(selected_paths: List[str], no_copy: bool) -> str:
    if not _active_world:
        return "Error: No active world selected. Please select a world first."
    if not selected_paths:
        return "Info: No files or folders selected to add."

    results = []
    success_count = 0
    error_count = 0

    all_files_to_process = []

    for selected_path_str in selected_paths:
        selected_path = Path(selected_path_str)
        if selected_path.is_file():
            all_files_to_process.append(selected_path)
        elif selected_path.is_dir():
            for item in selected_path.rglob('*'): # Recursively find all files
                if item.is_file():
                    all_files_to_process.append(item)
        else:
            results.append(f"Warning: Selected path '{selected_path_str}' is neither a file nor a directory. Skipping.")
            error_count += 1

    if not all_files_to_process:
        summary = f"Operation Summary: 0 succeeded, {error_count} failed (initial path checks).\n"
        return summary + "\n".join(results) + "\nNo valid files found to process."

    for filepath in all_files_to_process:
        try:
            if not filepath.exists() or not filepath.is_file(): # Double check, especially for symlinks etc.
                results.append(f"Error: File '{filepath.name}' (path: {str(filepath)}) not found or is not a valid file during processing.")
                error_count +=1
                continue

            original_filename, size_bytes, mime_type = file_operations.get_file_properties(filepath)
            event_type_str = "Reference (no copy)" if no_copy else "Copy"
            event_to_dispatch = (AssetReferenceIngestionRequested if no_copy else AssetFileIngestionRequested)(
                filepath_on_disk=filepath, original_filename=original_filename,
                mime_type=mime_type, size_bytes=size_bytes, world_name=_active_world.name,
            )
            await _active_world.dispatch_event(event_to_dispatch)
            # Consider if METADATA_EXTRACTION should be run per file or batched after all dispatches.
            # For now, keeping it per file as it was.
            await _active_world.execute_stage(dam.core.stages.SystemStage.METADATA_EXTRACTION)
            results.append(f"Success: Dispatched ingestion for '{original_filename}' (Type: {event_type_str}). Path: {filepath}")
            success_count += 1
        except Exception as e:
            results.append(f"Error processing file '{filepath.name}': {str(e)}")
            error_count += 1

    summary = f"Operation Summary: {success_count} succeeded, {error_count} failed.\n"
    return summary + "\n".join(results)

async def find_by_hash_ui(hash_value: str, hash_type: str) -> gr.JSON:
    if not _active_world:
        return gr.JSON(value={"error": "Error: No active world selected."}, label="Find by Hash Result")
    if not hash_value or not hash_value.strip():
        return gr.JSON(value={"error": "Error: Hash value cannot be empty."}, label="Find by Hash Result")

    request_id = str(uuid.uuid4())
    query_event = FindEntityByHashQuery(
        hash_value=hash_value.strip(), hash_type=hash_type.lower(),
        world_name=_active_world.name, request_id=request_id,
    )
    loop = asyncio.get_running_loop()
    query_event.result_future = loop.create_future()

    try:
        await _active_world.dispatch_event(query_event)
        details = await asyncio.wait_for(query_event.result_future, timeout=10.0)
        if details:
            return gr.JSON(value=details, label=f"Asset Found (Request ID: {request_id})")
        else:
            return gr.JSON(value={"info": f"Info: No asset found for hash {hash_value} (Type: {hash_type})."}, label="Find by Hash Result")
    except asyncio.TimeoutError:
        return gr.JSON(value={"error": f"Error: Query timed out for hash {hash_value}."}, label="Find by Hash Result")
    except Exception as e:
        return gr.JSON(value={"error": f"Error: Could not find asset by hash. {str(e)}"}, label="Find by Hash Result")

async def find_similar_images_ui(
    image_file_path: Optional[str], phash_threshold: int = 4,
    ahash_threshold: int = 4, dhash_threshold: int = 4
) -> gr.JSON:
    if not _active_world:
        return gr.JSON(value={"error": "Error: No active world selected."}, label="Similarity Search Result")
    if not image_file_path:
        return gr.JSON(value={"error": "Error: No image uploaded for similarity search."}, label="Similarity Search Result")

    image_path = Path(image_file_path)
    if not image_path.exists() or not image_path.is_file():
        return gr.JSON(value={"error": f"Error: Uploaded image file not found at path: {image_file_path}"}, label="Similarity Search Result")

    request_id = str(uuid.uuid4())
    query_event = FindSimilarImagesQuery(
        image_path=image_path, phash_threshold=phash_threshold, ahash_threshold=ahash_threshold,
        dhash_threshold=dhash_threshold, world_name=_active_world.name, request_id=request_id,
    )
    loop = asyncio.get_running_loop()
    query_event.result_future = loop.create_future()

    try:
        await _active_world.dispatch_event(query_event)
        results = await asyncio.wait_for(query_event.result_future, timeout=30.0)
        if results:
            if isinstance(results, list) and len(results) > 0 and "error" in results[0]:
                 return gr.JSON(value={"info": f"Info: {results[0]['error']}"}, label="Similarity Search Result")
            return gr.JSON(value=results, label=f"Similar Images Found (Request ID: {request_id})")
        else:
            return gr.JSON(value={"info": "Info: No similar images found."}, label="Similarity Search Result")
    except asyncio.TimeoutError:
        return gr.JSON(value={"error": "Error: Similarity query timed out."}, label="Similarity Search Result")
    except Exception as e:
        return gr.JSON(value={"error": f"Error: Could not find similar images. {str(e)}"}, label="Similarity Search Result")

async def get_transcode_profile_choices() -> list:
    if not _active_world:
        return ["Info: No world selected"]
    try:
        async with _active_world.get_db_session() as session:
            from dam.models.conceptual.transcode_profile_component import TranscodeProfileComponent
            from sqlalchemy import select
            stmt = select(TranscodeProfileComponent.profile_name).order_by(TranscodeProfileComponent.profile_name)
            result = await session.execute(stmt)
            profile_names = [row[0] for row in result.all() if row[0]]
            if not profile_names:
                return ["Info: No transcode profiles found"]
            return profile_names
    except Exception as e:
        return [f"Error: Could not load profiles. {str(e)}"]

async def transcode_asset_ui(asset_id_input: Optional[Union[str, int]], profile_name: Optional[str]) -> str:
    if not _active_world:
        return "Error: No active world selected."
    if not asset_id_input:
        return "Error: Asset ID must be provided."
    if not profile_name or profile_name.startswith("Info:") or profile_name.startswith("Error:") or profile_name == "No profiles found":
        return "Error: A valid transcode profile must be selected."

    try:
        source_asset_entity_id = int(asset_id_input)
        transcoded_entity = await transcode_service.apply_transcode_profile(
            world=_active_world, source_asset_entity_id=source_asset_entity_id,
            profile_entity_id_or_name=profile_name,
        )
        if transcoded_entity:
            return f"Success: Transcoded asset ID {source_asset_entity_id} using profile '{profile_name}'. New Asset ID: {transcoded_entity.id}."
        else:
            return f"Warning: Transcoding asset ID {source_asset_entity_id} with profile '{profile_name}' did not return a new asset. Check logs."
    except ValueError:
        return f"Error: Invalid Asset ID format '{asset_id_input}'. Must be an integer."
    except TranscodeServiceError as tse:
        return f"Error: Transcode Service Error. {str(tse)}"
    except FileNotFoundError as fnfe:
        return f"Error: File Not Found during transcoding. {str(fnfe)}"
    except Exception as e:
        return f"Error: An unexpected error occurred during transcoding. {str(e)}"

# Helper functions for manage_characters_ui
async def _resolve_asset_id(session, identifier: str) -> int:
    try:
        return int(identifier)
    except ValueError:
        resolved_id = await dam_ecs_service.find_entity_id_by_hash(session, identifier, "sha256")
        if not resolved_id:
            raise dam_ecs_service.EntityNotFoundError(f"Asset with identifier '{identifier}' (interpreted as hash) not found.")
        return resolved_id

async def _resolve_character_id(session, identifier: str) -> int:
    try:
        char_id = int(identifier)
        char_concept = await character_service.get_character_concept_by_id(session, char_id) # Verify ID exists
        if not char_concept:
             raise CharacterConceptNotFoundError(f"Character concept with ID '{identifier}' not found.")
        return char_id
    except ValueError: # Not an int, assume name
        char_concept = await character_service.get_character_concept_by_name(session, identifier)
        if not char_concept:
            raise CharacterConceptNotFoundError(f"Character concept with name '{identifier}' not found.")
        return char_concept.id

async def manage_characters_ui(
    action: str, character_identifier: Optional[str], asset_identifier: Optional[str],
    character_description: Optional[str], role_in_asset: Optional[str]
) -> str:
    if not _active_world:
        return "Error: No active world selected."

    character_identifier = character_identifier.strip() if character_identifier else None
    asset_identifier = asset_identifier.strip() if asset_identifier else None
    character_description = character_description.strip() if character_description else None
    role_in_asset = role_in_asset.strip() if role_in_asset else None

    try:
        async with _active_world.get_db_session() as session:
            if action == "create":
                if not character_identifier:
                    return "Error: Character name must be provided for creation."
                char_entity = await character_service.create_character_concept(
                    session=session, name=character_identifier, description=character_description
                )
                await session.commit()
                return f"Success: Character concept '{character_identifier}' (Entity ID: {char_entity.id}) created."

            elif action == "apply_to_asset":
                if not asset_identifier or not character_identifier:
                    return "Error: Both asset identifier and character identifier must be provided."
                asset_entity_id = await _resolve_asset_id(session, asset_identifier)
                char_concept_id = await _resolve_character_id(session, character_identifier)
                await character_service.apply_character_to_entity(
                    session=session, entity_id_to_link=asset_entity_id,
                    character_concept_entity_id=char_concept_id, role=role_in_asset
                )
                await session.commit()
                return f"Success: Character '{character_identifier}' applied to asset '{asset_identifier}' with role '{role_in_asset or ''}'."

            elif action == "list_for_asset":
                if not asset_identifier:
                    return "Error: Asset identifier must be provided."
                asset_entity_id = await _resolve_asset_id(session, asset_identifier)
                characters_on_asset = await character_service.get_characters_for_entity(session, asset_entity_id)
                if not characters_on_asset:
                    return f"Info: No characters found for asset '{asset_identifier}' (ID: {asset_entity_id})."
                output_lines = [f"Characters on asset '{asset_identifier}' (ID: {asset_entity_id}):"]
                for char_concept_entity, role in characters_on_asset:
                    char_comp = await dam_ecs_service.get_component(session, char_concept_entity.id, character_service.CharacterConceptComponent)
                    char_name = char_comp.concept_name if char_comp else "Unknown Character"
                    output_lines.append(f"  - {char_name} (Concept ID: {char_concept_entity.id}){' (Role: ' + role + ')' if role else ''}")
                return "\n".join(output_lines)

            elif action == "find_assets_for_character":
                if not character_identifier:
                    return "Error: Character identifier must be provided."
                char_concept_id = await _resolve_character_id(session, character_identifier)
                char_name_display = character_identifier
                char_comp_display = await dam_ecs_service.get_component(session, char_concept_id, character_service.CharacterConceptComponent)
                if char_comp_display: char_name_display = char_comp_display.concept_name
                linked_assets_info = await character_service.get_entities_for_character(
                    session, char_concept_id, role_filter=role_in_asset
                )
                if not linked_assets_info:
                    return f"Info: No assets found for character '{char_name_display}' (ID: {char_concept_id}) with role filter '{role_in_asset or 'Any'}'. "
                output_lines = [f"Assets for character '{char_name_display}' (ID: {char_concept_id}), role filter '{role_in_asset or 'Any'}':"]
                for asset_entity in linked_assets_info:
                    fpc = await dam_ecs_service.get_component(session, asset_entity.id, FilePropertiesComponent)
                    filename = fpc.original_filename if fpc else "N/A"
                    output_lines.append(f"  - Asset ID: {asset_entity.id}, Filename: {filename}")
                return "\n".join(output_lines)
            else:
                return f"Error: Unknown character management action '{action}'."
    except (CharacterConceptNotFoundError, dam_ecs_service.EntityNotFoundError, CharacterLinkNotFoundError) as e: # Corrected exception type
        return f"Error: {str(e)}"
    except ValueError as e:
        return f"Error: Invalid input. {str(e)}"
    except Exception as e:
        return f"Error: An unexpected error in character management. {str(e)}"

async def semantic_search_ui(query: str, top_n: int = 10, model_name: Optional[str] = None) -> gr.JSON:
    if not _active_world:
        return gr.JSON(value={"error": "Error: No active world selected."}, label="Semantic Search Results")
    if not query or not query.strip():
        return gr.JSON(value={"error": "Error: Search query cannot be empty."}, label="Semantic Search Results")

    request_id = str(uuid.uuid4())
    query_event = SemanticSearchQuery(
        query_text=query, world_name=_active_world.name, request_id=request_id, top_n=top_n,
        model_name=model_name.strip() if model_name and model_name.strip() else None,
    )
    loop = asyncio.get_running_loop()
    query_event.result_future = loop.create_future()

    try:
        await _active_world.dispatch_event(query_event)
        results = await asyncio.wait_for(query_event.result_future, timeout=60.0)
        if not results:
            return gr.JSON(value={"info": f"Info: No semantic matches found for query: '{query[:100]}...'"}, label="Semantic Search Results")
        formatted_results = []
        async with _active_world.get_db_session() as session:
            for entity, score, emb_comp in results:
                fpc = await dam_ecs_service.get_component(session, entity.id, FilePropertiesComponent)
                filename = fpc.original_filename if fpc else "N/A"
                formatted_results.append({
                    "entity_id": entity.id, "score": float(f"{score:.4f}"), "filename": filename,
                    "matched_on_source_component": emb_comp.source_component_name if emb_comp else "N/A",
                    "matched_on_field": emb_comp.source_field_name if emb_comp else "N/A",
                    "embedding_model": emb_comp.model_name if emb_comp else "N/A",
                })
        return gr.JSON(value=formatted_results, label=f"Semantic Search Results (Request ID: {request_id})")
    except asyncio.TimeoutError:
        return gr.JSON(value={"error": "Error: Semantic search query timed out."}, label="Semantic Search Results")
    except Exception as e:
        return gr.JSON(value={"error": f"Error: Semantic search failed. {str(e)}"}, label="Semantic Search Results")

async def create_evaluation_run_ui(run_name: str, description: Optional[str]) -> str:
    if not _active_world:
        return "Error: No active world selected."
    if not run_name or not run_name.strip():
        return "Error: Evaluation run name cannot be empty."
    description = description.strip() if description else None
    try:
        run_entity = await evaluation_systems.create_evaluation_run_concept(
            world=_active_world, run_name=run_name, description=description
        )
        return f"Success: Evaluation run '{run_name}' (Entity ID: {run_entity.id}) created."
    except EvaluationError as ee:
        return f"Error: {str(ee)}"
    except Exception as e:
        return f"Error: Could not create evaluation run. {str(e)}"

async def execute_evaluation_run_ui(
    run_identifier: str, asset_identifiers_str: str, profile_identifiers_str: str
) -> str:
    if not _active_world:
        return "Error: No active world selected."
    if not run_identifier or not run_identifier.strip() or \
       not asset_identifiers_str or not asset_identifiers_str.strip() or \
       not profile_identifiers_str or not profile_identifiers_str.strip():
        return "Error: Run identifier, asset identifiers, and profile identifiers must all be provided."

    asset_identifiers = [item.strip() for item in asset_identifiers_str.split(",") if item.strip()]
    profile_identifiers = [item.strip() for item in profile_identifiers_str.split(",") if item.strip()]

    if not asset_identifiers or not profile_identifiers:
        return "Error: Asset and profile identifier lists cannot be empty after stripping whitespace."

    typed_asset_ids: List[Union[int, str]] = []
    for aid in asset_identifiers:
        try: typed_asset_ids.append(int(aid))
        except ValueError: typed_asset_ids.append(aid)
    typed_profile_ids: List[Union[int, str]] = []
    for pid in profile_identifiers:
        try: typed_profile_ids.append(int(pid))
        except ValueError: typed_profile_ids.append(pid)

    run_id_to_use: Union[int, str]
    try: run_id_to_use = int(run_identifier)
    except ValueError: run_id_to_use = run_identifier

    try:
        results = await evaluation_systems.execute_evaluation_run(
            world=_active_world, evaluation_run_id_or_name=run_id_to_use,
            source_asset_identifiers=typed_asset_ids, profile_identifiers=typed_profile_ids,
        )
        return f"Success: Evaluation run '{run_identifier}' executed. Generated {len(results)} results. View report for details."
    except EvaluationError as ee:
        return f"Error: {str(ee)}"
    except Exception as e:
        return f"Error: Could not execute evaluation run. {str(e)}"

async def get_evaluation_run_choices() -> list:
    if not _active_world: return ["Info: No world selected"]
    try:
        async with _active_world.get_db_session() as session:
            from dam.models.conceptual.evaluation_run_component import EvaluationRunComponent
            from sqlalchemy import select
            stmt = select(EvaluationRunComponent.run_name).order_by(EvaluationRunComponent.run_name)
            result = await session.execute(stmt)
            run_names = [row[0] for row in result.all() if row[0]]
            if not run_names: return ["Info: No evaluation runs found"]
            return run_names
    except Exception as e: return [f"Error: Could not load runs. {str(e)}"]

async def get_evaluation_report_ui(run_identifier: str) -> gr.JSON:
    if not _active_world: return gr.JSON(value={"error": "Error: No world selected"})
    if not run_identifier or run_identifier.startswith("Info:") or run_identifier.startswith("Error:") or run_identifier == "No runs found":
        return gr.JSON(value={"error": "Error: Invalid run selected for report."})

    run_id_to_use: Union[int, str]
    try: run_id_to_use = int(run_identifier)
    except ValueError: run_id_to_use = run_identifier

    try:
        results_data = await evaluation_systems.get_evaluation_results(
            world=_active_world, evaluation_run_id_or_name=run_id_to_use
        )
        if not results_data:
            return gr.JSON(value={"info": f"Info: No results found for evaluation run '{run_identifier}'."})
        return gr.JSON(value=results_data)
    except EvaluationError as ee:
        return gr.JSON(value={"error": f"Error: {str(ee)}"})
    except Exception as e:
        return gr.JSON(value={"error": f"Error: Could not fetch report. {str(e)}"})

# --- World Operations ---
async def export_world_ui(export_path: str) -> str:
    if not _active_world:
        return "Error: No active world selected for export."
    if not export_path or not export_path.strip():
        return "Error: Export file path must be provided."

    export_file_path = Path(export_path.strip())
    if not export_file_path.is_absolute() and export_file_path.parent != Path("."): # Check parent if not current dir
        if not export_file_path.parent.exists():
            return f"Error: Parent directory '{str(export_file_path.parent)}' for export does not exist."
    if export_file_path.is_dir():
        return f"Error: Export path '{export_file_path}' is a directory, must be a file path."

    try:
        await asyncio.to_thread(world_service.export_ecs_world_to_json, _active_world, export_file_path)
        return f"Success: World '{_active_world.name}' exported to '{export_file_path}'."
    except Exception as e:
        return f"Error: Could not export world '{_active_world.name}'. {str(e)}"

async def import_world_ui(import_file_obj: Optional[gr.File], merge: bool) -> str:
    if not _active_world:
        return "Error: No active world selected for import."
    if import_file_obj is None or not hasattr(import_file_obj, 'name'):
        return "Error: Import file must be provided."

    import_file_path = Path(import_file_obj.name)
    if not import_file_path.exists() or not import_file_path.is_file():
        return f"Error: Uploaded import file not found at '{import_file_path}'."

    try:
        await asyncio.to_thread(world_service.import_ecs_world_from_json, _active_world, import_file_path, merge)
        return f"Success: Data imported from '{import_file_path.name}' into world '{_active_world.name}' (Merge: {merge})."
    except Exception as e:
        return f"Error: Could not import data into world '{_active_world.name}'. {str(e)}"

async def merge_worlds_ui(source_world_name: str, delete_source_after_merge: bool) -> str:
    if not _active_world:
        return "Error: No target world (current active world) selected for merge."
    if not source_world_name or source_world_name.startswith("Info:") or source_world_name.startswith("Error:"):
        return "Error: A valid source world must be selected."

    source_world_instance = get_current_world(source_world_name)
    if not source_world_instance:
        return f"Error: Source world '{source_world_name}' not found or could not be loaded."
    if _active_world.name == source_world_instance.name:
        return "Error: Target world and source world cannot be the same."

    try:
        await asyncio.to_thread(
            world_service.merge_ecs_worlds_db_to_db,
            source_world=source_world_instance, target_world=_active_world,
            strategy="add_new", delete_source_after_merge=delete_source_after_merge
        )
        return f"Success: Merged world '{source_world_name}' into '{_active_world.name}'. Delete source: {delete_source_after_merge}."
    except Exception as e:
        return f"Error: Could not merge worlds. {str(e)}"

async def split_world_ui(
    criteria_component_name: str, criteria_attribute_name: str, criteria_value: str,
    criteria_operator: str, selected_target_world_name: str,
    remaining_target_world_name: str, delete_from_source_after_split: bool
) -> str:
    if not _active_world:
        return "Error: No source world (current active world) selected for split."
    if not all(s and s.strip() for s in [criteria_component_name, criteria_attribute_name, criteria_value, criteria_operator,
                                          selected_target_world_name, remaining_target_world_name]):
        return "Error: All criteria fields and target world names must be provided and non-empty."

    selected_target_world_instance = get_current_world(selected_target_world_name)
    remaining_target_world_instance = get_current_world(remaining_target_world_name)

    if not selected_target_world_instance:
        return f"Error: Target world for selected entities '{selected_target_world_name}' not found."
    if not remaining_target_world_instance:
        return f"Error: Target world for remaining entities '{remaining_target_world_name}' not found."
    if _active_world.name == selected_target_world_name or \
       _active_world.name == remaining_target_world_name or \
       selected_target_world_name == remaining_target_world_name:
        return "Error: Source and both target worlds must be unique."

    try:
        # Ensure criteria_value is passed as-is; service layer should handle type conversion if needed.
        count_selected, count_remaining = await asyncio.to_thread(
            world_service.split_ecs_world,
            source_world=_active_world, target_world_selected=selected_target_world_instance,
            target_world_remaining=remaining_target_world_instance,
            criteria_component_name=criteria_component_name, criteria_component_attr=criteria_attribute_name,
            criteria_value=criteria_value, criteria_op=criteria_operator,
            delete_from_source=delete_from_source_after_split
        )
        return (f"Success: Split complete. {count_selected} entities to '{selected_target_world_name}', "
                f"{count_remaining} entities to '{remaining_target_world_name}'. Delete from source: {delete_from_source_after_split}.")
    except Exception as e:
        return f"Error: Could not split world. {str(e)}"

async def setup_db_ui() -> str:
    if not _active_world:
        return "Error: No active world selected to set up its database."
    try:
        await _active_world.create_db_and_tables()
        return f"Success: Database setup complete for world '{_active_world.name}'."
    except Exception as e:
        return f"Error: Could not set up database for world '{_active_world.name}'. {str(e)}"

# --- Gradio Interface Definition ---

# Helper async function for the .then() clause to ensure await is handled.
async def refresh_assets_default():
    """Calls list_assets_gr with default empty filters and page 1."""
    return await list_assets_gr(filename_filter="", mime_type_filter="", current_page=1)

def create_dam_ui():
    with gr.Blocks(title="ECS DAM System") as dam_interface:
        gr.Markdown("# ECS Digital Asset Management System")

        with gr.Row():
            world_selector = gr.Dropdown(choices=get_world_choices(), label="Select DAM World", value=get_world_choices()[0] if get_world_choices() else None, elem_id="world_selector")
            world_status_output = gr.Textbox(label="World Status", interactive=False, elem_id="world_status")

        confirm_world_button = gr.Button("Confirm World Selection & Refresh Dropdowns")

        with gr.Tabs() as main_tabs:
            with gr.TabItem("Assets"):
                asset_status_output = gr.Textbox(label="Asset List Status", interactive=False, elem_id="asset_status")
                with gr.Row():
                    filename_filter_input = gr.Textbox(label="Filter by Filename", elem_id="filename_filter")
                    mime_type_filter_input = gr.Dropdown(label="Filter by MIME Type", choices=[""], value="", elem_id="mime_type_filter")
                current_page_input = gr.Number(label="Page", value=1, minimum=1, step=1, interactive=True, elem_id="current_page")
                load_assets_button = gr.Button("Load/Refresh Assets")
                # Updated headers and col_count
                assets_df = gr.DataFrame(
                    headers=["ID", "Filename", "MIME Type", "File Size (Bytes)", "Created At"], # Updated header text
                    label="Assets",
                    interactive=True,
                    row_count=(20, "dynamic"),
                    col_count=(5,"fixed"),
                    elem_id="assets_dataframe"
                )
                asset_detail_json = gr.JSON(label="Asset Details (Tree View)", elem_id="asset_detail_json")

                load_assets_button.click(
                    list_assets_gr,
                    inputs=[filename_filter_input, mime_type_filter_input, current_page_input],
                    outputs=[assets_df, asset_status_output]
                )
                assets_df.select(get_asset_details_gr, inputs=[], outputs=asset_detail_json)

            with gr.TabItem("Operations"):
                with gr.Accordion("Add Assets", open=False):
                    # Configure FileExplorer: allow selection of files and directories.
                    # The root directory can be configured, e.g., to user's home or a specific data directory.
                    # For now, let it default or be configurable if Gradio supports it easily.
                    # glob parameter can be used to filter file types if needed, e.g. "*.jpg", "*.png"
                    # file_count="multiple" allows selecting multiple files/folders.
                    add_asset_files_input = gr.FileExplorer(label="Select Files or Folders to Add", file_count="multiple", root_dir="/", glob="*", elem_id="add_asset_files_explorer")
                    add_asset_no_copy_checkbox = gr.Checkbox(label="Add by reference (no copy)", value=False, elem_id="add_asset_no_copy")
                    add_asset_button = gr.Button("Add Selected Assets")
                    add_asset_output = gr.Textbox(label="Add Asset Status", lines=5, interactive=False, elem_id="add_asset_status")
                    add_asset_button.click(
                        add_assets_ui,
                        inputs=[add_asset_files_input, add_asset_no_copy_checkbox],
                        outputs=add_asset_output
                    )

                with gr.Accordion("Find Asset by Hash", open=False):
                    hash_value_input = gr.Textbox(label="Hash Value", elem_id="find_hash_value")
                    hash_type_dropdown = gr.Dropdown(choices=["md5", "sha256", "ahash", "phash", "dhash"], label="Hash Type", value="sha256", elem_id="find_hash_type")
                    find_hash_button = gr.Button("Find by Hash")
                    find_hash_output_json = gr.JSON(label="Find by Hash Result", elem_id="find_hash_output")
                    find_hash_button.click(
                        find_by_hash_ui,
                        inputs=[hash_value_input, hash_type_dropdown],
                        outputs=find_hash_output_json
                    )

                with gr.Accordion("Find Similar Images", open=False):
                    similar_image_upload_input = gr.Image(type="filepath", label="Upload Image for Similarity Search", elem_id="similar_image_upload")
                    with gr.Row():
                        phash_threshold_input = gr.Slider(minimum=0, maximum=64, value=4, step=1, label="pHash Threshold", elem_id="phash_thresh")
                        ahash_threshold_input = gr.Slider(minimum=0, maximum=64, value=4, step=1, label="aHash Threshold", elem_id="ahash_thresh")
                        dhash_threshold_input = gr.Slider(minimum=0, maximum=64, value=4, step=1, label="dHash Threshold", elem_id="dhash_thresh")
                    find_similar_button = gr.Button("Find Similar Images")
                    find_similar_output_json = gr.JSON(label="Similarity Search Result", elem_id="similar_image_output")
                    find_similar_button.click(
                        find_similar_images_ui,
                        inputs=[similar_image_upload_input, phash_threshold_input, ahash_threshold_input, dhash_threshold_input],
                        outputs=find_similar_output_json
                    )

                with gr.Accordion("Transcode Asset", open=False):
                    transcode_asset_id_input = gr.Number(label="Asset ID to Transcode", minimum=1, step=1, elem_id="transcode_asset_id")
                    transcode_profile_dropdown = gr.Dropdown(label="Transcode Profile", choices=["Info: Select world or refresh"], elem_id="transcode_profile_dropdown")
                    refresh_profiles_button = gr.Button("Refresh Transcode Profiles")
                    transcode_button = gr.Button("Transcode Asset")
                    transcode_output = gr.Textbox(label="Transcode Status", lines=3, interactive=False, elem_id="transcode_status")
                    refresh_profiles_button.click(get_transcode_profile_choices, [], transcode_profile_dropdown)
                    transcode_button.click(
                        transcode_asset_ui,
                        inputs=[transcode_asset_id_input, transcode_profile_dropdown],
                        outputs=transcode_output
                    )

                with gr.Accordion("Character Management", open=False):
                    with gr.Row():
                        char_action_dropdown = gr.Dropdown(
                            choices=["create", "apply_to_asset", "list_for_asset", "find_assets_for_character"],
                            label="Action", value="create", elem_id="char_action"
                        )
                    with gr.Row():
                        char_identifier_input = gr.Textbox(label="Character Name/ID", info="Name for create/find; Name or ID for apply/list.", elem_id="char_id")
                        char_description_input = gr.Textbox(label="Character Description (for create)", elem_id="char_desc")
                    with gr.Row():
                        asset_identifier_input = gr.Textbox(label="Asset ID/Hash (for apply/list)", info="Entity ID or SHA256 hash.", elem_id="char_asset_id")
                        char_role_input = gr.Textbox(label="Role (for apply/find_assets filter)", elem_id="char_role")
                    char_manage_button = gr.Button("Execute Character Action")
                    char_manage_output = gr.Textbox(label="Character Management Result", lines=5, interactive=False, elem_id="char_manage_status")
                    char_manage_button.click(
                        manage_characters_ui,
                        inputs=[char_action_dropdown, char_identifier_input, asset_identifier_input, char_description_input, char_role_input],
                        outputs=char_manage_output
                    )

                with gr.Accordion("Semantic Search", open=False):
                    semantic_query_input = gr.Textbox(label="Semantic Search Query", elem_id="semantic_query")
                    semantic_top_n_input = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Number of Results (Top N)", elem_id="semantic_top_n")
                    semantic_model_input = gr.Textbox(label="Embedding Model Name (optional, uses default if empty)", elem_id="semantic_model")
                    semantic_search_button = gr.Button("Search Semantically")
                    semantic_search_output_json = gr.JSON(label="Semantic Search Results", elem_id="semantic_search_output")
                    semantic_search_button.click(
                        semantic_search_ui,
                        inputs=[semantic_query_input, semantic_top_n_input, semantic_model_input],
                        outputs=semantic_search_output_json
                    )

                with gr.Accordion("Transcoding Evaluation", open=False):
                    gr.Markdown("### Create New Evaluation Run")
                    with gr.Row():
                        eval_run_name_input = gr.Textbox(label="New Run Name", elem_id="eval_run_name")
                        eval_run_desc_input = gr.Textbox(label="Description (optional)", elem_id="eval_run_desc")
                    create_eval_run_button = gr.Button("Create Evaluation Run")
                    create_eval_run_output = gr.Textbox(label="Create Run Status", interactive=False, elem_id="create_eval_run_status")
                    create_eval_run_button.click(
                        create_evaluation_run_ui, inputs=[eval_run_name_input, eval_run_desc_input], outputs=create_eval_run_output
                    )
                    gr.Markdown("### Execute Evaluation Run")
                    with gr.Row():
                        eval_run_identifier_dropdown = gr.Dropdown(label="Select Evaluation Run", choices=["Info: Select world or refresh"], elem_id="eval_run_id_execute")
                        refresh_eval_runs_button_execute = gr.Button("Refresh Run List")
                    eval_asset_ids_input = gr.Textbox(label="Asset IDs/Hashes (comma-separated)", elem_id="eval_asset_ids")
                    eval_profile_ids_input = gr.Textbox(label="Profile Names/IDs (comma-separated)", elem_id="eval_profile_ids")
                    execute_eval_run_button = gr.Button("Execute Evaluation Run")
                    execute_eval_run_output = gr.Textbox(label="Execute Run Status", interactive=False, lines=3, elem_id="execute_eval_run_status")
                    refresh_eval_runs_button_execute.click(get_evaluation_run_choices, [], eval_run_identifier_dropdown)
                    execute_eval_run_button.click(
                        execute_evaluation_run_ui,
                        inputs=[eval_run_identifier_dropdown, eval_asset_ids_input, eval_profile_ids_input],
                        outputs=execute_eval_run_output
                    )
                    gr.Markdown("### View Evaluation Report")
                    with gr.Row():
                        eval_run_report_dropdown = gr.Dropdown(label="Select Evaluation Run for Report", choices=["Info: Select world or refresh"], elem_id="eval_run_id_report")
                        refresh_eval_runs_button_report = gr.Button("Refresh Run List")
                    view_eval_report_button = gr.Button("View Report")
                    eval_report_output_json = gr.JSON(label="Evaluation Report", elem_id="eval_report_output")
                    refresh_eval_runs_button_report.click(get_evaluation_run_choices, [], eval_run_report_dropdown)
                    view_eval_report_button.click(
                        get_evaluation_report_ui, inputs=[eval_run_report_dropdown], outputs=eval_report_output_json
                    )

            with gr.TabItem("World Management"):
                with gr.Accordion("Export Current World", open=False):
                    export_path_input = gr.Textbox(label="Export File Path (e.g., /path/to/export.json)", elem_id="world_export_path")
                    export_world_button = gr.Button("Export World")
                    export_world_output = gr.Textbox(label="Export Status", interactive=False, elem_id="world_export_status")
                    export_world_button.click(export_world_ui, inputs=[export_path_input], outputs=export_world_output)

                with gr.Accordion("Import Data into Current World", open=False):
                    import_file_input = gr.File(label="Select Import File (.json)", type="filepath", elem_id="world_import_file")
                    import_merge_checkbox = gr.Checkbox(label="Merge with existing data", value=False, elem_id="world_import_merge")
                    import_world_button = gr.Button("Import Data")
                    import_world_output = gr.Textbox(label="Import Status", interactive=False, elem_id="world_import_status")
                    import_world_button.click(import_world_ui, inputs=[import_file_input, import_merge_checkbox], outputs=import_world_output)

                with gr.Accordion("Merge Worlds", open=False):
                    all_world_names = get_world_choices()
                    source_world_merge_dropdown = gr.Dropdown(choices=all_world_names, label="Source World to Merge From", elem_id="world_merge_source")
                    delete_source_merge_checkbox = gr.Checkbox(label="Delete Source World After Merge", value=False, elem_id="world_merge_delete_source")
                    merge_worlds_button = gr.Button("Merge Worlds")
                    merge_worlds_output = gr.Textbox(label="Merge Status", interactive=False, elem_id="world_merge_status")
                    merge_worlds_button.click(
                        merge_worlds_ui, inputs=[source_world_merge_dropdown, delete_source_merge_checkbox], outputs=merge_worlds_output
                    )

                with gr.Accordion("Split Current World", open=False):
                    gr.Markdown("Criteria for selecting entities to move to 'Selected Target World':")
                    split_criteria_component_input = gr.Textbox(label="Component Name (e.g., FilePropertiesComponent)", elem_id="world_split_comp_name")
                    split_criteria_attribute_input = gr.Textbox(label="Attribute Name (e.g., mime_type)", elem_id="world_split_attr_name")
                    split_criteria_value_input = gr.Textbox(label="Attribute Value (e.g., image/jpeg)", elem_id="world_split_attr_val")
                    split_criteria_operator_dropdown = gr.Dropdown(
                        choices=["eq", "ne", "lt", "le", "gt", "ge", "like", "ilike", "contains", "startswith", "endswith"],
                        label="Operator", value="eq", elem_id="world_split_op")
                    gr.Markdown("Target worlds (must be existing configured worlds different from source):")
                    all_world_names_for_split = get_world_choices()
                    selected_target_world_dropdown = gr.Dropdown(choices=all_world_names_for_split, label="Selected Target World Name", elem_id="world_split_target_selected")
                    remaining_target_world_dropdown = gr.Dropdown(choices=all_world_names_for_split, label="Remaining Target World Name", elem_id="world_split_target_remaining")
                    delete_from_source_split_checkbox = gr.Checkbox(label="Delete Entities from Source After Split", value=False, elem_id="world_split_delete_source")
                    split_world_button = gr.Button("Split World")
                    split_world_output = gr.Textbox(label="Split Status", interactive=False, elem_id="world_split_status")
                    split_world_button.click(
                        split_world_ui,
                        inputs=[split_criteria_component_input, split_criteria_attribute_input,
                                split_criteria_value_input, split_criteria_operator_dropdown,
                                selected_target_world_dropdown, remaining_target_world_dropdown,
                                delete_from_source_split_checkbox],
                        outputs=split_world_output
                    )

                with gr.Accordion("Setup Database for Current World", open=False):
                    setup_db_button = gr.Button("Initialize/Setup Database for Current Active World")
                    setup_db_output = gr.Textbox(label="Database Setup Status", interactive=False, elem_id="world_setup_db_status")
                    setup_db_button.click(setup_db_ui, inputs=[], outputs=setup_db_output)

        confirm_world_button.click(
            set_active_world_and_refresh_dropdowns,
            inputs=[world_selector],
            outputs=[
                world_status_output,
                mime_type_filter_input,
                transcode_profile_dropdown,
                eval_run_identifier_dropdown,
                eval_run_report_dropdown
            ]
        ).then(
            refresh_assets_default, # Replaced lambda with the async helper function
            inputs=[],
            outputs=[assets_df, asset_status_output]
        )
    return dam_interface

if __name__ == "__main__":
    print("Attempting to initialize worlds for standalone Gradio UI...")
    try:
        from dam.core import config as app_config
        # Changed import for get_world_by_name
        from dam.core.world import get_world as get_world_by_name, create_and_register_all_worlds_from_settings
        from dam.core.world_setup import register_core_systems

        initialized_worlds = create_and_register_all_worlds_from_settings(app_settings=app_config.settings)
        for world_instance in initialized_worlds:
            register_core_systems(world_instance)

        if app_config.settings.DEFAULT_WORLD_NAME:
            _active_world = get_world_by_name(app_config.settings.DEFAULT_WORLD_NAME)
            print(f"Standalone: Default world '{_active_world.name if _active_world else 'None'}' pre-selected.")
        elif initialized_worlds:
            _active_world = initialized_worlds[0]
            print(f"Standalone: First initialized world '{_active_world.name if _active_world else 'None'}' pre-selected.")
        else:
            print("Standalone: No worlds configured or could be initialized.")
            if not any(w.name == "default" for w in app_settings.worlds): # Check if already added by some other logic
                 from dam.core.world import World
                 dummy_world = World(name="default", db_url="sqlite+aiosqlite:///:memory:") # Use a distinct in-memory DB
                 app_settings.worlds.append(dummy_world)
                 register_core_systems(dummy_world) # Also register systems for the dummy world
                 _active_world = dummy_world
                 print("Standalone: Created a dummy 'default' world for UI testing.")
    except Exception as e:
        print(f"Standalone: Error initializing worlds: {str(e)}")
        print("Ensure your .env or config file is set up correctly if you expect worlds to load.")
        if not _active_world and (not hasattr(app_settings, 'worlds') or not app_settings.worlds or not any(w.name == "default" for w in app_settings.worlds)):
            from dam.core.world import World
            _active_world = World(name="default", db_url="sqlite+aiosqlite:///:memory:_standalone") # Use a distinct in-memory DB
            if not hasattr(app_settings, 'worlds') or not app_settings.worlds: app_settings.worlds = []
            app_settings.worlds.append(_active_world)
            register_core_systems(_active_world)
            print("Standalone: Created a fallback dummy 'default' world due to init errors.")

    import logging
    from dam.core.logging_config import setup_logging
    setup_logging(level=logging.INFO)

    ui = create_dam_ui()
    ui.launch()

def launch_ui(world_name: Optional[str] = None):
    global _active_world
    if world_name:
        from dam.core.world import get_world as get_world_by_name # Changed import
        _active_world = get_world_by_name(world_name)
    elif app_settings.worlds:
        _active_world = app_settings.worlds[0]

    # If still no active world (e.g. no config, no default passed) try to init for UI launch
    if not _active_world:
        print("UI Launch: No active world. Attempting to initialize from settings for UI...")
        try:
            initialized_worlds = create_and_register_all_worlds_from_settings(app_settings=app_settings)
            for world_instance in initialized_worlds:
                register_core_systems(world_instance)
            if app_settings.DEFAULT_WORLD_NAME:
                _active_world = get_world_by_name(app_settings.DEFAULT_WORLD_NAME)
            elif initialized_worlds:
                _active_world = initialized_worlds[0]

            if _active_world:
                 print(f"UI Launch: Using world '{_active_world.name}'")
            else:
                 print("UI Launch: Still no active world after attempting init. UI may be limited.")
        except Exception as e:
            print(f"UI Launch: Error during fallback world initialization: {str(e)}")


    interface = create_dam_ui()
    interface.launch()
