from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from pfun_cma_model.misc.pathdefs import PFunDataPaths


def setup_admin_backend():
    """Setup the admin database and sqlalchemy engine."""
    Base = declarative_base()
    pfun_dpaths = PFunDataPaths()
    pfun_dpaths.ensure_local_share_path_exists()
    engine = create_async_engine(
        pfun_dpaths.admin_db_fpath,
        connect_args={"check_same_thread": False},
    )
    Session = sessionmaker(bind=engine, class_=AsyncSession)  # type: ignore
    return Base, engine, Session


Base, engine, Session = setup_admin_backend()
