"""WebSocket handler for proposal events."""

from uuid import UUID

from infrastructure.communication.http.websocket.enums.websocket_event_type import (
    WebSocketEventType,
)
from infrastructure.communication.http.websocket.services.websocket_manager import (
    WebSocketManager,
)
from infrastructure.communication.http.websocket.types.execution_result_data import (
    ExecutionResultData,
)
from infrastructure.communication.http.websocket.types.proposal_justification_data import (
    ProposalJustificationData,
)
from infrastructure.communication.http.websocket.types.proposal_update_data import (
    ProposalUpdateData,
)


class ProposalWebSocketHandler:
    """Handler for proposal-related WebSocket events."""

    def __init__(self, websocket_manager: WebSocketManager):
        """Initialize with WebSocket manager."""
        self.websocket_manager = websocket_manager

    async def notify_proposal_created(
        self,
        project_id: UUID,
        proposal_id: UUID,
        agent_id: UUID,
        proposal_type: str,
        content: str,
        justification: ProposalJustificationData,
    ) -> None:
        """Notify subscribers about new proposal."""
        proposal_data = ProposalUpdateData(
            agent_id=str(agent_id),
            proposal_type=proposal_type,
            content=content,
            justification=justification,
            status="PENDING",
        )

        await self.websocket_manager.send_proposal_update(
            project_id=project_id,
            proposal_id=proposal_id,
            event_type=WebSocketEventType.PROPOSAL_CREATED.value,
            proposal_data=proposal_data,
        )

    async def notify_proposal_approved(
        self, project_id: UUID, proposal_id: UUID, approved_by: UUID
    ) -> None:
        """Notify subscribers about approved proposal."""
        proposal_data = ProposalUpdateData(
            agent_id="",
            proposal_type="",
            content="",
            justification=ProposalJustificationData(),
            status="APPROVED",
            approved_by=str(approved_by),
        )

        await self.websocket_manager.send_proposal_update(
            project_id=project_id,
            proposal_id=proposal_id,
            event_type=WebSocketEventType.PROPOSAL_APPROVED.value,
            proposal_data=proposal_data,
        )

    async def notify_proposal_rejected(
        self, project_id: UUID, proposal_id: UUID
    ) -> None:
        """Notify subscribers about rejected proposal."""
        proposal_data = ProposalUpdateData(
            agent_id="",
            proposal_type="",
            content="",
            justification=ProposalJustificationData(),
            status="REJECTED",
        )

        await self.websocket_manager.send_proposal_update(
            project_id=project_id,
            proposal_id=proposal_id,
            event_type=WebSocketEventType.PROPOSAL_REJECTED.value,
            proposal_data=proposal_data,
        )

    async def notify_proposal_executed(
        self,
        project_id: UUID,
        proposal_id: UUID,
        execution_result: ExecutionResultData,
    ) -> None:
        """Notify subscribers about executed proposal."""
        proposal_data = ProposalUpdateData(
            agent_id="",
            proposal_type="",
            content="",
            justification=ProposalJustificationData(),
            status="EXECUTED",
            execution_result=execution_result,
        )

        await self.websocket_manager.send_proposal_update(
            project_id=project_id,
            proposal_id=proposal_id,
            event_type=WebSocketEventType.PROPOSAL_EXECUTED.value,
            proposal_data=proposal_data,
        )

    async def notify_proposal_failed(
        self, project_id: UUID, proposal_id: UUID, error_message: str
    ) -> None:
        """Notify subscribers about failed proposal execution."""
        proposal_data = ProposalUpdateData(
            agent_id="",
            proposal_type="",
            content="",
            justification=ProposalJustificationData(),
            status="FAILED",
            error_message=error_message,
        )

        await self.websocket_manager.send_proposal_update(
            project_id=project_id,
            proposal_id=proposal_id,
            event_type=WebSocketEventType.PROPOSAL_FAILED.value,
            proposal_data=proposal_data,
        )
