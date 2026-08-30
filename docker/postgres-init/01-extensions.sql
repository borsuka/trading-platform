-- Extensions the application relies on.
--
-- pg_stat_statements is enabled because diagnosing a slow trading loop after the fact is
-- otherwise guesswork.
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
