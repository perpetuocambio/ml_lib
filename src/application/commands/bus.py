"""Command Bus implementation.

Central dispatcher for commands with middleware support.
"""

import logging
from typing import Type
from ml_lib.diffusion.application.commands.base import (
    ICommand,
    ICommandHandler,
    ICommandBus,
    CommandResult,
    CommandStatus,
)

logger = logging.getLogger(__name__)


class CommandBus(ICommandBus):
    """
    Simple command bus implementation.

    Features:
    - Handler registration by command type
    - Command dispatching to appropriate handler
    - Error handling and logging
    - Extensible via middleware (future)

    Example:
        bus = CommandBus()
        bus.register(MyCommand, MyCommandHandler(dependencies))
        result = bus.dispatch(MyCommand(param1="value"))
    """

    def __init__(self):
        """Initialize empty command bus."""
        self._handlers: dict[Type[ICommand], ICommandHandler] = {}
        logger.info("CommandBus initialized")

    def register(self, command_type: Type[ICommand], handler: ICommandHandler) -> None:
        """
        Register handler for command type.

        Args:
            command_type: Command class
            handler: Handler instance

        Raises:
            ValueError: If handler already registered for command type
        """
        if command_type in self._handlers:
            raise ValueError(
                f"Handler already registered for {command_type.__name__}"
            )

        self._handlers[command_type] = handler
        logger.debug(f"Registered handler for {command_type.__name__}")

    def dispatch(self, command: ICommand) -> CommandResult:
        """
        Dispatch command to appropriate handler.

        Args:
            command: Command to execute

        Returns:
            CommandResult from handler

        Raises:
            ValueError: If no handler registered for command type
        """
        command_type = type(command)
        command_name = command_type.__name__

        # Find handler
        handler = self._handlers.get(command_type)
        if handler is None:
            error_msg = f"No handler registered for {command_name}"
            logger.error(error_msg)
            return CommandResult.failure(error_msg)

        # Execute handler
        try:
            logger.debug(f"Dispatching {command_name}")
            result = handler.handle(command)

            if result.is_success:
                logger.debug(f"{command_name} executed successfully")
            else:
                logger.warning(
                    f"{command_name} failed: {result.status.name} - {result.error}"
                )

            return result

        except Exception as e:
            error_msg = f"Exception in {command_name} handler: {str(e)}"
            logger.exception(error_msg)
            return CommandResult.failure(error_msg, CommandStatus.FAILED)

    def is_registered(self, command_type: Type[ICommand]) -> bool:
        """
        Check if handler is registered for command type.

        Args:
            command_type: Command class

        Returns:
            True if handler registered, False otherwise
        """
        return command_type in self._handlers

    def unregister(self, command_type: Type[ICommand]) -> None:
        """
        Unregister handler for command type.

        Args:
            command_type: Command class
        """
        if command_type in self._handlers:
            del self._handlers[command_type]
            logger.debug(f"Unregistered handler for {command_type.__name__}")

    def get_registered_commands(self) -> list[Type[ICommand]]:
        """
        Get list of registered command types.

        Returns:
            List of command types with registered handlers
        """
        return list(self._handlers.keys())
