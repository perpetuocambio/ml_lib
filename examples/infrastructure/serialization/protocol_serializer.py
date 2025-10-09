"""Protocol serializer for infrastructure boundaries - ONLY place with dict usage in Infrastructure."""

from typing import Any, TypeVar

from infrastructure.config.types.raw_config_data import RawConfigData
from infrastructure.persistence.database.config.database_config_entity import (
    DatabaseConfig,
)

# Generic type for dataclass serialization
T = TypeVar("T")


class ProtocolSerializer:
    """Handles conversion from typed classes to dict format for infrastructure boundaries ONLY."""

    @staticmethod
    def serialize_config_data(config_data: RawConfigData) -> dict[str, Any]:
        """Convert typed config data to dict for file/environment boundaries ONLY."""
        # Infrastructure boundary: convert typed structure to dict for config files
        if hasattr(config_data, "to_dict"):
            return config_data.to_dict()

        # Handle dataclass instances
        if hasattr(config_data, "__dataclass_fields__"):
            result = {}
            for field_name in config_data.__dataclass_fields__:
                value = getattr(config_data, field_name)
                if hasattr(value, "to_protocol_value"):
                    result[field_name] = value.to_protocol_value()
                else:
                    result[field_name] = value
            return result

        # Fallback for simple types
        return config_data if isinstance(config_data, dict) else {}

    @staticmethod
    def deserialize_config_data(
        data: dict[str, Any], target_class: type[RawConfigData]
    ) -> RawConfigData:
        """Convert dict from config files to typed config data."""
        # Infrastructure boundary: convert dict from config files to typed structure
        if hasattr(target_class, "from_dict"):
            return target_class.from_dict(data)

        # Handle dataclass construction
        if hasattr(target_class, "__dataclass_fields__"):
            field_values = {}
            for field_name, _field in target_class.__dataclass_fields__.items():
                if field_name in data:
                    field_values[field_name] = data[field_name]
            return target_class(**field_values)

        return data

    @staticmethod
    def serialize_database_row(entity: T) -> dict[str, Any]:
        """Convert dataclass entity to dict for database ORM boundaries ONLY."""
        # Infrastructure boundary: convert entity to dict for database persistence
        if hasattr(entity, "to_db_dict"):
            return entity.to_db_dict()

        # Handle dataclass instances
        if hasattr(entity, "__dataclass_fields__"):
            result = {}
            for field_name in entity.__dataclass_fields__:
                value = getattr(entity, field_name)
                # Convert complex types to database-compatible formats
                if hasattr(value, "to_db_value"):
                    result[field_name] = value.to_db_value()
                else:
                    result[field_name] = value
            return result

        return {}

    @staticmethod
    def deserialize_database_row(
        data: dict[str, Any], entity_class: type[DatabaseConfig]
    ) -> DatabaseConfig:
        """Convert database row dict to domain entity."""
        # Infrastructure boundary: convert dict from database to domain entity
        if hasattr(entity_class, "from_db_dict"):
            return entity_class.from_db_dict(data)

        # Handle dataclass construction
        if hasattr(entity_class, "__dataclass_fields__"):
            field_values = {}
            for field_name, _field in entity_class.__dataclass_fields__.items():
                if field_name in data:
                    field_values[field_name] = data[field_name]
            return entity_class(**field_values)

        return data

    @staticmethod
    def serialize_http_payload(payload: Any) -> dict[str, Any]:
        """Convert typed payload to dict for HTTP boundaries ONLY."""
        # Infrastructure boundary: convert typed structure to dict for HTTP requests
        if hasattr(payload, "to_http_dict"):
            return payload.to_http_dict()

        # Handle dataclass instances
        if hasattr(payload, "__dataclass_fields__"):
            result = {}
            for field_name in payload.__dataclass_fields__:
                value = getattr(payload, field_name)
                if hasattr(value, "to_protocol_value"):
                    result[field_name] = value.to_protocol_value()
                else:
                    result[field_name] = value
            return result

        return {}

    @staticmethod
    def deserialize_http_payload(data: dict[str, Any], payload_class: type[Any]) -> Any:
        """Convert HTTP response dict to typed payload."""
        # Infrastructure boundary: convert dict from HTTP response to typed structure
        if hasattr(payload_class, "from_http_dict"):
            return payload_class.from_http_dict(data)

        # Handle dataclass construction
        if hasattr(payload_class, "__dataclass_fields__"):
            field_values = {}
            for field_name, _field in payload_class.__dataclass_fields__.items():
                if field_name in data:
                    field_values[field_name] = data[field_name]
            return payload_class(**field_values)

        return data

    @staticmethod
    def serialize_dict_data(data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Convert dict data to protocol-safe format for infrastructure boundaries ONLY."""
        # Infrastructure boundary: ensure dict is properly serialized for protocol use
        if data is None:
            return {}
        return dict(data)

    @staticmethod
    def serialize_mapping_data(mapping: dict[Any, Any]) -> dict[str, Any]:
        """Convert mapping data to protocol-safe string-keyed format for infrastructure boundaries ONLY."""
        # Infrastructure boundary: convert any mapping to string-keyed dict for protocol use
        result: dict[str, Any] = {}
        for key, value in mapping.items():
            str_key = str(key) if not isinstance(key, str) else key
            result[str_key] = value
        return result
