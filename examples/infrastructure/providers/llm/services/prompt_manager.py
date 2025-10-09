"""
Centralized prompt management system for agent prompts.
Loads and manages templates from YAML configuration files.
Uses existing infrastructure: TypedYamlLoader.
"""

# Infrastructure layer cannot import from application layer
# TypedYamlLoader will be injected as dependency
from infrastructure.config.prompts.entities.prompts_config import PromptsConfig
from infrastructure.providers.llm.entities.proposal_template_fields import (
    ProposalTemplateFields,
)
from infrastructure.providers.llm.entities.template_categories import (
    TemplateCategoriesListing,
)


class PromptManager:
    """
    Centralizes prompt template management for agents.

    Provides template rendering with variable substitution.
    """

    def __init__(self, prompts_config: PromptsConfig):
        """Initialize prompt manager with loaded configuration."""
        self._prompts_config = prompts_config

    def get_system_prompt(self, template_id: str, **variables) -> str:
        """
        Get rendered system prompt for an agent role.

        Args:
            template_id: ID of the prompt template
            **variables: Variables to substitute in template

        Returns:
            Rendered prompt string

        Raises:
            KeyError: If template_id not found
        """
        if (
            not self._prompts_config
            or template_id not in self._prompts_config.system_prompts
        ):
            raise KeyError(f"System prompt template not found: {template_id}")

        template_config = self._prompts_config.system_prompts[template_id]

        # Validate required variables
        missing_vars = [
            var for var in template_config.variables if var not in variables
        ]
        if missing_vars:
            raise ValueError(
                f"Missing required variables for {template_id}: {missing_vars}"
            )

        # Render template with variables
        return self._render_template(template_config.template, **variables)

    def get_reasoning_prompt(self, prompt_type: str, **variables) -> str:
        """
        Get rendered reasoning prompt for agent decision-making.

        Args:
            prompt_type: Type of reasoning prompt
            **variables: Variables to substitute in template

        Returns:
            Rendered prompt string
        """
        if (
            not self._prompts_config
            or prompt_type not in self._prompts_config.reasoning_prompts
        ):
            raise KeyError(f"Reasoning prompt not found: {prompt_type}")

        template_config = self._prompts_config.reasoning_prompts[prompt_type]

        return self._render_template(template_config.template, **variables)

    def get_proposal_template(
        self, action_type: str, **variables
    ) -> ProposalTemplateFields:
        """
        Get proposal template for a specific action type.

        Args:
            action_type: Type of action for the proposal
            **variables: Variables to substitute in template

        Returns:
            ProposalTemplateFields with rendered template fields
        """
        if (
            not self._prompts_config
            or action_type not in self._prompts_config.proposal_templates
        ):
            return ProposalTemplateFields(
                description=f"Ejecutar {action_type}",
                expected_outcome=f"Resultado de {action_type}",
            )

        template = self._prompts_config.proposal_templates[action_type]

        return ProposalTemplateFields(
            description=self._render_template(
                template.get("description", f"Ejecutar {action_type}"), **variables
            ),
            expected_outcome=self._render_template(
                template.get("expected_outcome", f"Resultado de {action_type}"),
                **variables,
            ),
        )

    def _render_template(self, template: str, **variables) -> str:
        """
        Render template string with variable substitution.

        Args:
            template: Template string with {{variable}} placeholders
            **variables: Variables to substitute

        Returns:
            Rendered string
        """
        result = template

        for key, value in variables.items():
            placeholder = f"{{{{{key}}}}}"  # {{variable}} format

            # Convert lists to formatted strings
            if isinstance(value, list):
                if key.endswith("_list"):
                    if key == "goals_list":
                        value = "\n".join([f"- {goal}" for goal in value])
                    elif key == "available_tools_list":
                        value = "\n".join([f"- {tool}" for tool in value])
                    elif key == "specializations_list":
                        value = ", ".join(value)
                    elif key == "observations_list":
                        value = "\n".join([f"- {obs}" for obs in value])
                    else:
                        value = ", ".join(str(item) for item in value)
                else:
                    value = ", ".join(str(item) for item in value)

            result = result.replace(placeholder, str(value))

        return result

    def reload_prompts(self, prompts_config: PromptsConfig) -> None:
        """Reload prompt templates from configuration."""
        self._prompts_config = prompts_config

    def list_available_templates(self) -> TemplateCategoriesListing:
        """
        List all available prompt templates.

        Returns:
            TemplateCategoriesListing with template categories and their IDs
        """
        if not self._prompts_config:
            return TemplateCategoriesListing(
                system_prompts=[], reasoning_prompts=[], proposal_templates=[]
            )

        return TemplateCategoriesListing(
            system_prompts=list(self._prompts_config.system_prompts.keys()),
            reasoning_prompts=list(self._prompts_config.reasoning_prompts.keys()),
            proposal_templates=list(self._prompts_config.proposal_templates.keys()),
        )
