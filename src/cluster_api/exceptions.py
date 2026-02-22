class IdeaNotFoundError(Exception):
    """Raised when an idea ID does not exist in the database."""

    def __init__(self, idea_id: int):
        super().__init__(f"Idea {idea_id} not found")
        self.idea_id = idea_id


class AlreadyClusteredError(Exception):
    """Raised when an idea has already been clustered with a given method."""

    def __init__(self, idea_id: int, method: str):
        super().__init__(f"Idea {idea_id} already clustered with {method}")
        self.idea_id = idea_id
        self.method = method
