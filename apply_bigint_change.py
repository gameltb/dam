from sqlalchemy import create_engine, text
from dam.core.config import settings

def main():
    db_url = settings.get_world_config('default').DATABASE_URL
    engine = create_engine(db_url)
    with engine.connect() as connection:
        connection.execute(text("ALTER TABLE component_file_properties ALTER COLUMN file_size_bytes TYPE BIGINT;"))
        connection.commit()
    print("Successfully altered component_file_properties table.")

if __name__ == "__main__":
    main()
