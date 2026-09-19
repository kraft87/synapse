"""Database facade combining stores on one transaction-aware connection."""

from ingestion.db_connection import _EMBED_DIMS as _EMBED_DIMS
from ingestion.db_connection import _vector_literal as _vector_literal
from ingestion.db_documents import DocumentStore
from ingestion.db_episodes import EpisodeStore
from ingestion.db_notes import NoteStore
from ingestion.db_preferences import PreferenceStore
from ingestion.db_queue import ExtractionQueueStore
from ingestion.db_timeline import TimelineStore


class Database(
    EpisodeStore, TimelineStore, PreferenceStore, NoteStore, DocumentStore, ExtractionQueueStore
):
    """Public store API; each instance owns one synchronous database connection."""
