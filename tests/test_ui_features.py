@pytest.mark.ui # Ensure it's marked as a UI test
def test_main_window_populate_mime_type_filter_success(main_window_with_mocks: MainWindow, mock_world, mocker, qtbot: QtBot):
    """Test successful population of the MIME type filter in MainWindow."""
    main_window = main_window_with_mocks
    mock_mime_types = ["image/jpeg", "image/png", "application/pdf"]

    # Configure the mock_db_session_instance (provided by mock_world) for this specific test's needs.
    # This overrides the default empty list behavior set in mock_world fixture.
    mock_execution_result = mocker.MagicMock() # This is the object returned by 'await session.execute()'

    # This is the object returned by 'result.scalars()' (synchronous call)
    mock_scalar_result_object = mocker.MagicMock()
    mock_scalar_result_object.all.return_value = mock_mime_types # .all() is sync, returns the list

    # Configure mock_execution_result.scalars (the method) to return mock_scalar_result_object
    mock_execution_result.scalars = mocker.MagicMock(return_value=mock_scalar_result_object)

    # Configure the execute method on the session instance provided by mock_world
    mock_world.mock_db_session_instance.execute = mocker.AsyncMock(return_value=mock_execution_result)

    # MainWindow.__init__ (called by fixture) already triggered populate_mime_type_filter
    # which used the default mock_world DB settings (empty results).
    # We need to re-trigger populate_mime_type_filter to use the new mock setup for *this test*.
    main_window.populate_mime_type_filter() # Explicitly call with new DB mock behavior

    main_window.show() # Ensure window is shown for UI updates
    qtbot.waitForWindowShown(main_window)

    def check_mime_filter_populated():
        return main_window.mime_type_filter.count() > 1 # Check for more than just "All Types"
    qtbot.waitUntil(check_mime_filter_populated, timeout=5000)

    assert main_window.mime_type_filter.count() == len(mock_mime_types) + 1
    # Items in combobox are sorted alphabetically due to MimeTypeFetcher query and _update_mime_type_filter_ui logic
    sorted_mock_mime_types = sorted(list(set(mock_mime_types)))
    for i, mime_type in enumerate(sorted_mock_mime_types):
        assert main_window.mime_type_filter.itemText(i + 1) == mime_type # +1 to skip "All Types"
        assert main_window.mime_type_filter.itemData(i + 1) == mime_type
    assert main_window.mime_type_filter.isEnabled()
