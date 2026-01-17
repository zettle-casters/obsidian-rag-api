from neomodel.config import get_config

def init_db(db_url : str) -> None:
    config = get_config()
    config.database_url = db_url
    config.soft_cardinality_check = True
