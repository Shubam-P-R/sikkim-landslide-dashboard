"""
=============================================================================
seed_db.py — Neon PostgreSQL Database Seeder
=============================================================================
Author  : Backend Data Engineer
Tasks   :
  1. Connect to Neon PostgreSQL via SQLAlchemy + psycopg2
  2. Read sikkim_landslide_cleaned.csv
  3. Create & populate  -> location_features  (1,073 rows)
  4. Create spatial index -> idx_coords (LATITUDE, LONGITUDE)
  5. Create              -> incident_reports  (user contributions)
=============================================================================
"""

import sys
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
CONNECTION_STRING = (
    "postgresql+psycopg2://neondb_owner:npg_eu3W1bYFTdqL"
    "@ep-empty-tree-avfhcwdo-pooler.c-11.us-east-1.aws.neon.tech"
    "/neondb?sslmode=require&channel_binding=require"
)

CSV_PATH   = Path("sikkim_landslide_cleaned.csv")
TABLE_NAME = "location_features"
BATCH_SIZE = 500          # rows per insert batch

DIVIDER = "=" * 60

def section(title: str) -> None:
    print(f"\n{DIVIDER}\n  {title}\n{DIVIDER}")

# ---------------------------------------------------------------------------
# 2. Load CSV
# ---------------------------------------------------------------------------
section("LOADING CSV")

if not CSV_PATH.exists():
    sys.exit(f"[ERROR] CSV not found: {CSV_PATH.resolve()}")

df = pd.read_csv(CSV_PATH)
print(f"  Rows loaded   : {len(df):,}")
print(f"  Columns       : {list(df.columns)}")

# Cast to correct dtypes to match DB schema
dtype_map = {
    "LONGITUDE" : float,
    "LATITUDE"  : float,
    "CLASS"     : int,
    "ASPECT1"   : int,
    "ELEVATION1": float,
    "GEOLOGY1"  : float,
    "LULC1"     : float,
    "NDVI1"     : float,
    "RAINFALL1" : float,
    "ROAD1"     : float,
    "SLOPE1"    : int,
    "SOIL1"     : float,
    "STREAM1"   : float,
}
for col, dtype in dtype_map.items():
    if col in df.columns:
        df[col] = df[col].astype(dtype)

print(f"  Dtypes applied: OK")

# ---------------------------------------------------------------------------
# 3. Connect to Neon PostgreSQL
# ---------------------------------------------------------------------------
section("CONNECTING TO NEON POSTGRESQL")

try:
    engine = create_engine(
        CONNECTION_STRING,
        connect_args={"connect_timeout": 15},
        pool_pre_ping=True,
    )
    with engine.connect() as conn:
        result = conn.execute(text("SELECT version();"))
        ver = result.fetchone()[0]
    print(f"  Connected!  PostgreSQL: {ver[:60]}...")
except SQLAlchemyError as e:
    sys.exit(f"[ERROR] Connection failed:\n{e}")

# ---------------------------------------------------------------------------
# 4. Create & Populate location_features
# ---------------------------------------------------------------------------
section(f"CREATING TABLE: {TABLE_NAME}")

create_table_sql = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id          SERIAL PRIMARY KEY,
    "LONGITUDE"  FLOAT,
    "LATITUDE"   FLOAT,
    "CLASS"      INTEGER,
    "ASPECT1"    INTEGER,
    "ELEVATION1" FLOAT,
    "GEOLOGY1"   FLOAT,
    "LULC1"      FLOAT,
    "NDVI1"      FLOAT,
    "RAINFALL1"  FLOAT,
    "ROAD1"      FLOAT,
    "SLOPE1"     INTEGER,
    "SOIL1"      FLOAT,
    "STREAM1"    FLOAT
);
"""

with engine.begin() as conn:
    conn.execute(text(f'DROP TABLE IF EXISTS {TABLE_NAME} CASCADE;'))
    conn.execute(text(create_table_sql))
    print(f"  Table '{TABLE_NAME}' created (or replaced).")

# Batch-insert using pandas to_sql
df.to_sql(
    TABLE_NAME,
    engine,
    if_exists="append",
    index=False,
    method="multi",
    chunksize=BATCH_SIZE,
)

# Verify row count
with engine.connect() as conn:
    count = conn.execute(text(f'SELECT COUNT(*) FROM {TABLE_NAME};')).scalar()
print(f"  Rows inserted : {count:,}  (expected 1,073)")

# ---------------------------------------------------------------------------
# 5. Spatial Composite Index
# ---------------------------------------------------------------------------
section("CREATING SPATIAL INDEX")

with engine.begin() as conn:
    conn.execute(text(
        f'CREATE INDEX IF NOT EXISTS idx_coords '
        f'ON {TABLE_NAME} ("LATITUDE", "LONGITUDE");'
    ))
print('  Index created : idx_coords  ON location_features ("LATITUDE", "LONGITUDE")')

# ---------------------------------------------------------------------------
# 6. Create incident_reports Table
# ---------------------------------------------------------------------------
section("CREATING TABLE: incident_reports")

create_incidents_sql = """
CREATE TABLE IF NOT EXISTS incident_reports (
    id          SERIAL PRIMARY KEY,
    latitude    FLOAT NOT NULL,
    longitude   FLOAT NOT NULL,
    description TEXT,
    severity    VARCHAR(50),
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

with engine.begin() as conn:
    conn.execute(text(create_incidents_sql))

# Verify columns
with engine.connect() as conn:
    cols = conn.execute(text("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'incident_reports'
        ORDER BY ordinal_position;
    """)).fetchall()

print("  Table 'incident_reports' ready.")
print("  Columns:")
for col_name, col_type in cols:
    print(f"    {col_name:<15} {col_type}")

# ---------------------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------------------
section("SEEDING COMPLETE")
print(f"""
  Tables created
  ──────────────
  location_features   {count:>6,} rows   (landslide feature data)
  incident_reports         0 rows   (awaiting user reports)

  Indexes
  ───────
  idx_coords  ON location_features (LATITUDE, LONGITUDE)

  Connection
  ──────────
  Host : ep-empty-tree-avfhcwdo-pooler.c-11.us-east-1.aws.neon.tech
  DB   : neondb
  SSL  : required
""")
