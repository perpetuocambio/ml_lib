"""Negative prompt utilities for content safety and quality control."""

from typing import Dict, List, Set
import re
from ml_lib.diffusion.intelligent.prompting.config_loader import get_default_config


class NegativePromptGenerator:
    """Generates negative prompts based on content analysis and safety requirements."""
    
    def __init__(self, config=None):
        """
        Initialize negative prompt generator.
        
        Args:
            config: PrompterConfig with negative prompt definitions (if None, loads default)
        """
        if config is None:
            config = get_default_config()
        self.config = config
        self.negative_prompts = config.negative_prompts
        
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
                               safety_level: str = "strict") -> List[str]:
        """
        Generate negative prompt based on content type and safety requirements.
        
        Args:
            is_explicit: Whether content is explicit
            content_type: Type of content (photorealistic, anime, etc.)
            safety_level: Safety level ('strict', 'moderate', 'relaxed')
            
        Returns:
            List of negative prompt terms
        """
        negative_terms = set()
        
        # Add base negative prompts for content type
        if content_type in self.negative_prompts:
            negative_terms.update(self.negative_prompts[content_type])
        
        # Add general negative prompts
        if "general" in self.negative_prompts:
            negative_terms.update(self.negative_prompts["general"])
        
        # Add age inappropriate terms based on safety level
        age_inappropriate = self.negative_prompts.get("age_inappropriate", [])
        if safety_level in ["strict", "moderate"]:
            negative_terms.update(age_inappropriate)
        
        # Add specific explicit content filters if not explicit content
        if not is_explicit and "explicit" in self.negative_prompts:
            negative_terms.update(self.negative_prompts["explicit"])
            
        return list(negative_terms)
    
    def get_age_safe_negative_prompt(self) -> List[str]:
        """Get negative prompts specifically for age safety."""
        return self.negative_prompts.get("age_inappropriate", [])
    
    def should_block_prompt(self, prompt: str, threshold: float = 0.5) -> tuple[bool, float, List[str]]:
        """
        Determine if a prompt should be blocked based on safety threshold.
        
        Args:
            prompt: Prompt to check
            threshold: Safety score threshold (0.0-1.0). Below this = blocked.
            
        Returns:
            Tuple of (should_block, safety_score, violations)
        """
        safety_score = self.get_contextual_safety_score(prompt)
        is_safe, violations = self.is_prompt_safe(prompt)
        
        should_block = safety_score < threshold
        return should_block, safety_score, violations
    
    def is_prompt_safe(self, prompt: str, check_age_only: bool = False) -> tuple[bool, List[str]]:
        """
        Check if a prompt is safe based on negative terms with contextual analysis.
        
        Args:
            prompt: Prompt to check
            check_age_only: Only check for age-inappropriate content
            
        Returns:
            Tuple of (is_safe, list_of_violations)
        """
        prompt_lower = prompt.lower()
        violations = []
        
        # Basic pattern matching
        if check_age_only:
            age_inappropriate = self.negative_prompts.get("age_inappropriate", [])
            basic_violations = [term for term in age_inappropriate if term.lower() in prompt_lower]
        else:
            basic_violations = []
            # Check all negative prompt categories
            for category, terms in self.negative_prompts.items():
                for term in terms:
                    if term.lower() in prompt_lower:
                        basic_violations.append(term)
        
        # Contextual analysis for relationship terms
        context_violations = self._analyze_contextual_violations(prompt_lower)
        
        violations = basic_violations + context_violations
        
        # Apply contextual filtering only if there are potential violations
        if violations:
            filtered_violations = self._apply_contextual_filtering(violations, prompt_lower)
            return len(filtered_violations) == 0, filtered_violations
        
        return len(violations) == 0, violations
    
    def _analyze_contextual_violations(self, prompt_lower: str) -> List[str]:
        """Analyze potentially problematic terms with context."""
        violations = []
        
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
    
    def _apply_contextual_filtering(self, violations: List[str], prompt_lower: str) -> List[str]:
        """Apply contextual filtering to determine if violations are real."""
        filtered = []
        
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
        is_safe, violations = self.is_prompt_safe(prompt)
        
        if not violations:
            return 1.0
        
        # Calculate based on severity of violations
        age_violations = [v for v in violations if any(x in v for x in ['age_restricted', 'minor', 'child', 'teen'])]
        relation_violations = [v for v in violations if 'relation' in v]
        other_violations = [v for v in violations if v not in age_violations and v not in relation_violations]
        
        # Heuristic scoring: age violations are most severe
        score = 1.0
        score -= len(age_violations) * 0.4  # Severe: 0.4 per violation
        score -= len(relation_violations) * 0.2  # Moderate: 0.2 per violation  
        score -= len(other_violations) * 0.1  # Less severe: 0.1 per violation
        
        return max(0.0, min(1.0, score))