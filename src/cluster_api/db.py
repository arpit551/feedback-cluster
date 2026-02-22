from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Integer, LargeBinary, String, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker


class Base(DeclarativeBase):
    pass


class Idea(Base):
    __tablename__ = "ideas"

    id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column(String, nullable=False)
    user_id = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    embedding = Column(LargeBinary, nullable=True)

    clusters = relationship("IdeaCluster", back_populates="idea")


class Cluster(Base):
    __tablename__ = "clusters"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    method = Column(String, nullable=False)  # "bertopic" or "llm"
    centroid = Column(LargeBinary, nullable=True)
    size = Column(Integer, default=0)

    ideas = relationship("IdeaCluster", back_populates="cluster", cascade="all, delete-orphan")


class IdeaCluster(Base):
    __tablename__ = "idea_clusters"

    idea_id = Column(Integer, ForeignKey("ideas.id"), primary_key=True)
    cluster_id = Column(Integer, ForeignKey("clusters.id", ondelete="CASCADE"), primary_key=True)

    idea = relationship("Idea", back_populates="clusters")
    cluster = relationship("Cluster", back_populates="ideas")


_engine = None
_SessionLocal = None


def init_db(db_path: str = "ideas.db") -> None:
    global _engine, _SessionLocal
    _engine = create_engine(
        f"sqlite:///{db_path}",
        echo=False,
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(_engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    _SessionLocal = sessionmaker(bind=_engine)
    Base.metadata.create_all(_engine)


def get_session() -> Session:
    if _SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _SessionLocal()
