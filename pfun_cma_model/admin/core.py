from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, create_engine
from pfun_cma_model.misc.pathdefs import PFunDataPaths


def setup_admin_backend():
    """Setup the admin database and sqlalchemy engine."""
    Base = declarative_base()
    pfun_dpaths = PFunDataPaths()
    pfun_dpaths.ensure_local_share_path_exists()
    engine = create_engine(
        pfun_dpaths.admin_db_fpath,
        connect_args={"check_same_thread": False},
    )
    return Base, engine


Base, engine = setup_admin_backend()
