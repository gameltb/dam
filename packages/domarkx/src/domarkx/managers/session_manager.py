"""Manages the lifecycle of sessions."""

from domarkx.data.models import Session


class SessionManager:
    """A central service for creating, retrieving, and updating sessions."""

    def __init__(self) -> None:
        """Initialize the SessionManager."""
        self._sessions: dict[str, Session] = {}

    def create_session(self, session_id: str) -> Session:
        """
        Create a new session.

        Args:
            session_id (str): The ID of the session to create.

        Returns:
            Session: The newly created session.

        """
        if session_id in self._sessions:
            raise ValueError(f"Session with ID '{session_id}' already exists.")
        session = Session(session_id=session_id)
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Session | None:
        """
        Get a session by its ID.

        Args:
            session_id (str): The ID of the session to retrieve.

        Returns:
            Session | None: The session, or None if it does not exist.

        """
        return self._sessions.get(session_id)

    def update_session(self, session: Session) -> None:
        """
        Update a session.

        Args:
            session (Session): The session to update.

        """
        if session.session_id not in self._sessions:
            raise ValueError(f"Session with ID '{session.session_id}' does not exist.")
        self._sessions[session.session_id] = session
