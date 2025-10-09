"""Database configuration data types."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DatabaseConfigData:
    """Type-safe container for database configuration data - replaces dict with typed classes."""

    database_type: str
    connection_string: str
    min_pool_size: int
    max_pool_size: int
    command_timeout: int
    connection_timeout: int
    retry_attempts: int

    def mask_sensitive_data(self) -> "DatabaseConfigData":
        """Return copy with sensitive data masked."""
        # Mask password in connection string
        masked_conn_str = self.connection_string
        if "password=" in masked_conn_str.lower():
            masked_conn_str = re.sub(
                r"password=[^;]*", "password=***", masked_conn_str, flags=re.IGNORECASE
            )

        return DatabaseConfigData(
            database_type=self.database_type,
            connection_string=masked_conn_str,
            min_pool_size=self.min_pool_size,
            max_pool_size=self.max_pool_size,
            command_timeout=self.command_timeout,
            connection_timeout=self.connection_timeout,
            retry_attempts=self.retry_attempts,
        )
