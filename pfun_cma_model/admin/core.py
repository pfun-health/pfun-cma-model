import os
from passlib.context import CryptContext
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


# Initialize the admin backend (database, engine, session maker)
Base, engine, Session = setup_admin_backend()


def setup_pwd_context() -> CryptContext:
    """Setup the password context for hashing and verifying passwords."""

    # Initialize password context for hashing and verifying passwords
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    pwd_context.load_path(
        os.path.join(PFunDataPaths._pfun_root_path, "SECURITY_POLICY.ini")
    )
    return pwd_context


# Initialize the password context for hashing and verifying passwords
pwd_context = setup_pwd_context()
