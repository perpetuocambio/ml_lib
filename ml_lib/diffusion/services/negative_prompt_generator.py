"""Negative prompt utilities for content safety and quality control."""

from typing import Optional
import re

from ml_lib.diffusion.models.value_objects import (
    PromptBlockResult,
    PromptSafetyResult,
)


class NegativePromptGenerator:
    """Generates negative prompts based on content analysis and safety requirements."""
    
    def __init__(self, config: Optional[object] = None):
        """
        Initialize negative prompt generator.

        Args:
            config: Optional object with negative_prompts attribute (if None, uses defaults)
        """
        # Define default negative prompts as explicit attributes
        self._general_negatives = ['low quality', 'blurry', 'deformed', 'bad anatomy']
        self._photorealistic_negatives = ['cartoon', 'anime', 'unrealistic']
        self._age_inappropriate_negatives = ['child', 'minor', 'teen', 'underage']
        self._explicit_negatives = ['nsfw']

        self.config = config
        
        # Define context-sensitive terms that require additional analysis
        self.relation_terms = [
            'daughter', 'granddaughter', 'sister', 'cousin', 'niece', 
            'stepdaughter', 'adopted daughter', 'child'
        ]
        
        # Terms that are ALWAYS blocked regardless of context (highest priority)
        self.always_blocked = [
            'baby', 'infant', 'toddler', 'loli', 'shota', 'child', 'minor'
        ]
        
        # Terms that indicate adult content (when used with relationship terms)
        self.adult_indicators = [
            'milf', 'mature', 'older', '40s', '50s', '60s', 'experienced', 
            'grown', 'adult', 'mature woman', 'senior', 'older woman'
        ]
        
        # Age-restricted terms (should be blocked regardless of context)
        self.age_restricted = [
            'teen', 'teenager', 'underage', 'minor', 'child', 'kid', 'infant', 
            'baby', 'toddler', '18', '19', '20s', 'under 18', 'below 18'
        ]
        
        # Medical/condition terms that are acceptable when applied to adults
        self.acceptable_medical = [
            'dwarf', 'little person', 'dwarfism', 'height', 'petite', 'small',
            'little', 'compact', 'short stature', 'proportions'
        ]
        
    def generate_negative_prompt(self,
                               is_explicit: bool = True,
                               content_type: str = "photorealistic",
                               safety_level: str = "strict") -> list[str]:
        """
        Generate negative prompt based on content type and safety requirements.

        Args:
            is_explicit: Whether content is explicit
            content_type: Type of content (photorealistic, anime, etc.)
            safety_level: Safety level ('strict', 'moderate', 'relaxed')

        Returns:
            List of negative prompt terms
        """
        negative_terms: set[str] = set()

        # Add base negative prompts for content type
        if content_type == "photorealistic":
            negative_terms.update(self._photorealistic_negatives)

        # Add general negative prompts
        negative_terms.update(self._general_negatives)

        # Add age inappropriate terms based on safety level
        if safety_level in ["strict", "moderate"]:
            negative_terms.update(self._age_inappropriate_negatives)

        # Add specific explicit content filters if not explicit content
        if not is_explicit:
            negative_terms.update(self._explicit_negatives)

        return list(negative_terms)

    def get_age_safe_negative_prompt(self) -> list[str]:
        """Get negative prompts specifically for age safety."""
        return self._age_inappropriate_negatives.copy()
    
    def should_block_prompt(self, prompt: str, threshold: float = 0.5) -> PromptBlockResult:
        """
        Determine if a prompt should be blocked based on safety threshold.

        Args:
            prompt: Prompt to check
            threshold: Safety score threshold (0.0-1.0). Below this = blocked.

        Returns:
            PromptBlockResult with block decision, score, and violations
        """
        safety_score = self.get_contextual_safety_score(prompt)
        safety_result = self.is_prompt_safe(prompt)

        should_block = safety_score < threshold
        return PromptBlockResult(
            should_block=should_block,
            safety_score=safety_score,
            violations=safety_result.violations
        )

    def is_prompt_safe(self, prompt: str, check_age_only: bool = False) -> PromptSafetyResult:
        """
        Check if a prompt is safe based on negative terms with contextual analysis.

        Args:
            prompt: Prompt to check
            check_age_only: Only check for age-inappropriate content

        Returns:
            PromptSafetyResult with safety status and violations
        """
        prompt_lower = prompt.lower()
        violations: list[str] = []

        # Basic pattern matching
        if check_age_only:
            basic_violations = [
                term for term in self._age_inappropriate_negatives
                if term.lower() in prompt_lower
            ]
        else:
            basic_violations = []
            # Check all negative prompt categories
            all_negatives = (
                self._general_negatives +
                self._photorealistic_negatives +
                self._age_inappropriate_negatives +
                self._explicit_negatives
            )
            for term in all_negatives:
                if term.lower() in prompt_lower:
                    basic_violations.append(term)

        # Contextual analysis for relationship terms
        context_violations = self._analyze_contextual_violations(prompt_lower)

        violations = basic_violations + context_violations

        # Apply contextual filtering only if there are potential violations
        if violations:
            filtered_violations = self._apply_contextual_filtering(violations, prompt_lower)
            is_safe = len(filtered_violations) == 0
            return PromptSafetyResult(is_safe=is_safe, violations=filtered_violations)

        is_safe = len(violations) == 0
        return PromptSafetyResult(is_safe=is_safe, violations=violations)
    
    def _analyze_contextual_violations(self, prompt_lower: str) -> list[str]:
        """Analyze potentially problematic terms with context."""
        violations: list[str] = []
        
        # Check for terms that are ALWAYS blocked (highest priority)
        for term in self.always_blocked:
            if term in prompt_lower:
                violations.append(f"always_blocked:{term}")
        
        # Check for age-restricted terms (these are always blocked)
        for term in self.age_restricted:
            if term in prompt_lower:
                violations.append(f"age_restricted:{term}")
        
        # Analyze relationship terms with age context
        for relation in self.relation_terms:
            if relation in prompt_lower:
                # Check if there are adult indicators nearby (within 10 words)
                words = prompt_lower.split()
                relation_idx = -1
                for i, word in enumerate(words):
                    if relation in word:
                        relation_idx = i
                        break
                
                if relation_idx != -1:
                    # Look for adult indicators near the relation term
                    start = max(0, relation_idx - 10)
                    end = min(len(words), relation_idx + 10)
                    context = ' '.join(words[start:end])
                    
                    # If no adult indicators AND relation term exists, flag it
                    has_adult_indicator = any(indicator in context for indicator in self.adult_indicators)
                    if not has_adult_indicator:
                        violations.append(f"relation_no_adult_context:{relation}")
        
        return violations
    
    def _apply_contextual_filtering(self, violations: list[str], prompt_lower: str) -> list[str]:
        """Apply contextual filtering to determine if violations are real."""
        filtered: list[str] = []
        
        for violation in violations:
            # Terms that are ALWAYS blocked (highest priority) - never filtered out
            if violation.startswith('always_blocked:'):
                filtered.append(violation)
                continue
            
            # Check if this is a medical/condition term that might be OK in adult context
            if any(term in violation for term in ['dwarf', 'little person', 'petite', 'small']):
                # If adult indicators are present, this might be OK
                if any(indicator in prompt_lower for indicator in self.adult_indicators):
                    continue  # Skip this violation
            
            # Check if this is age-restricted
            if violation.startswith('age_restricted:'):
                filtered.append(violation)
            # Check if this is relation without adult context
            elif violation.startswith('relation_no_adult_context:'):
                # This is potentially problematic but needs review of context
                rel_term = violation.split(':')[1]
                # Double-check if there are clear adult indicators elsewhere
                if any(indicator in prompt_lower for indicator in ['40s', '50s', '60s', 'milf', 'mature', 'older']):
                    continue  # Skip this violation
                else:
                    filtered.append(violation)
            # Regular violations
            else:
                filtered.append(violation)
        
        return filtered
    
    def get_contextual_safety_score(self, prompt: str) -> float:
        """
        Get a safety score (0.0-1.0) where 1.0 is completely safe.
        
        Args:
            prompt: Prompt to analyze
            
        Returns:
            Safety score between 0.0 (unsafe) and 1.0 (safe)
        """
        safety_result = self.is_prompt_safe(prompt)

        if not safety_result.violations:
            return 1.0

        # Calculate based on severity of violations
        violations = safety_result.violations
        age_violations = [v for v in violations if any(x in v for x in ['age_restricted', 'minor', 'child', 'teen'])]
        relation_violations = [v for v in violations if 'relation' in v]
        other_violations = [v for v in violations if v not in age_violations and v not in relation_violations]
        
        # Heuristic scoring: age violations are most severe
        score = 1.0
        score -= len(age_violations) * 0.4  # Severe: 0.4 per violation
        score -= len(relation_violations) * 0.2  # Moderate: 0.2 per violation  
        score -= len(other_violations) * 0.1  # Less severe: 0.1 per violation
        
        return max(0.0, min(1.0, score))