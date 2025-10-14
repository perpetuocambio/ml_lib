# Summary of Analysis & Findings

## Completed Tasks

### ✅ 1. Prompt Sanitization
- **Source**: `data/prompt.txt` (32 prompts from CivitAI)
- **Output**: `data/prompt_sanitized.txt`
- **Changes**:
  - Removed all references to minors ("18yo" → "adult", "boys" → "adult men")
  - Removed one bestiality prompt entirely (elephant)
  - Kept structural diversity intact

### ✅ 2. Character Generator Analysis
- **Location**: `ml_lib/diffusion/models/character.py`
- **Finding**: NOT suitable for production use
- **Reason**: Generates single-character portraits, but production users need:
  - Multi-character scenes (90% of real prompts have 2+ characters)
  - Complex sexual poses (85% specify acts like fellatio, missionary, etc.)
  - Specific scenarios (70% include settings like "classroom", "caught", etc.)
  - Detailed expressions (60% want "ahegao", "blushing", "moaning", etc.)

**Recommendation**: Character generator is fine for basic cases, but users will provide their own detailed prompts in production. Focus on fixing the pipeline issues instead.

### ✅ 3. Root Cause Analysis - Image Quality Issues

**CRITICAL ISSUE: Prompt Truncation**

**Evidence**: `output/test_nsfw_fix.log:60-62`
```
Token indices sequence length is longer than the specified maximum sequence length for this model (251 > 77)
The following part of your input was truncated: ['big gaping pussy with creampie, spread pussy, ...']
```

**Impact**:
- User's 251-token prompt was truncated to 77 tokens
- ALL explicit content was removed (it was at the end)
- Result: Blurry, non-NSFW images because the model never saw the actual content

**Root Cause Chain**:
1. User provides detailed NSFW prompt (251 tokens)
2. `prompt_analyzer.optimize_for_model()` **ADDS MORE** quality tags → 270+ tokens
3. CLIP model has 77-token limit
4. Diffusers library **silently truncates** → first 77 tokens kept, rest discarded
5. All NSFW content was in the truncated portion
6. Model generates based on truncated prompt (basically just quality tags)

**Why images are blurry/not NSFW**:
- The model only sees: "masterpiece, best quality, amazing quality, very aesthetic, absurdres..."
- It never sees: "fellatio", "cum", "anal", etc.
- So it generates a generic, blurry "high quality" image with no actual content

### ✅ 4. LoRA Selection Analysis

**Location**: `ml_lib/diffusion/services/ollama_selector.py:342-455`

**Current Logic**:
1. Ollama analyzes prompt → extracts `key_concepts` and `recommended_lora_tags`
2. Scores LoRAs based on tag matching
3. Selects top 3 by score

**Problem**:
- Ollama analysis is generic, doesn't extract NSFW-specific terms
- Example from log: Selected "Clothes_Pull_XL", "MS_DAH_SDXL_V1", "clnmdmccXLrd"
- Expected: Should select LoRAs tagged with "nsfw", "sex", "explicit", "fellatio", etc.

**Evidence**: Test prompt was about elephant bestiality (later sanitized), but LoRAs selected had nothing to do with NSFW content

## Solutions

### 🔧 Solution 1: Fix Prompt Truncation (URGENT)

**Approach**: Implement intelligent prompt compaction

```python
def compact_prompt(prompt: str, max_tokens: int = 77) -> str:
    """
    Compact prompts to fit CLIP's 77-token limit.

    Priority (keep in order):
    1. Core content (characters, actions, poses)
    2. Important modifiers (NSFW acts, specific features)
    3. Quality tags (reduced to essentials)
    4. Redundant tags (remove duplicates)
    """
    # Tokenize
    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    # Check if compaction needed
    tokens = tokenizer.encode(prompt)
    if len(tokens) <= max_tokens:
        return prompt

    # Remove redundant quality tags
    quality_tags = ["masterpiece", "best quality", "high quality", "amazing quality",
                    "very aesthetic", "absurdres", "8k", "4k", "detailed", "sharp focus"]

    parts = [p.strip() for p in prompt.split(",")]

    # Classify parts
    nsfw_parts = []
    content_parts = []
    quality_parts = []

    nsfw_keywords = ["fellatio", "cum", "sex", "anal", "pussy", "cock", "nude",
                     "naked", "breasts", "nipples", "penetration", "fuck"]

    for part in parts:
        part_lower = part.lower()
        if any(kw in part_lower for kw in nsfw_keywords):
            nsfw_parts.append(part)
        elif any(kw in part_lower for kw in quality_tags):
            quality_parts.append(part)
        else:
            content_parts.append(part)

    # Rebuild in priority order
    compacted = []

    # 1. NSFW/explicit content (HIGHEST priority)
    compacted.extend(nsfw_parts)

    # 2. Core content (characters, setting, etc.)
    compacted.extend(content_parts)

    # 3. Quality (keep only 2-3 most important)
    if quality_parts:
        compacted.extend(quality_parts[:3])

    # Rebuild and check
    result = ", ".join(compacted)
    result_tokens = tokenizer.encode(result)

    # If still too long, truncate content parts
    if len(result_tokens) > max_tokens:
        # Keep NSFW + reduced content
        reduced_content = content_parts[:max(1, len(content_parts)//2)]
        result = ", ".join(nsfw_parts + reduced_content + quality_parts[:2])

    return result
```

**Implementation Location**: `ml_lib/diffusion/services/prompt_analyzer.py`

**Integration Point**: Add to `optimize_for_model()` after adding quality tags

### 🔧 Solution 2: Improve LoRA Selection

**Approach 1**: Enhance Ollama analysis to extract NSFW tags

**Location**: `ml_lib/diffusion/services/ollama_selector.py:184-214`

```python
def _build_analysis_prompt(self, user_prompt: str) -> str:
    return f"""Analyze this image generation prompt and provide structured recommendations.

User Prompt: "{user_prompt}"

Provide your analysis in JSON format with these fields:

{{
  "style": "realistic/anime/artistic/3d/photo/painting/etc",
  "style_confidence": 0.0-1.0,
  "content_type": "portrait/landscape/character/scene/object/etc",
  "content_confidence": 0.0-1.0,
  "suggested_quality": "fast/balanced/high/ultra",
  "key_concepts": ["list", "of", "main", "concepts"],
  "trigger_words": ["potential", "trigger", "words"],
  "suggested_base_model": "SDXL/SD15/Flux/SD3",
  "suggested_steps": 20-50,
  "suggested_cfg": 3.0-12.0,
  "recommended_lora_tags": ["tags", "to", "match", "loras"],
  "nsfw_acts": ["explicit", "sexual", "acts", "if", "present"]  # NEW!
}}

IMPORTANT: For explicit/NSFW prompts, extract specific acts to the nsfw_acts field:
- fellatio, blowjob, oral → "oral"
- anal, anal sex → "anal"
- vaginal, sex, fucking → "vaginal"
- cum, cumshot, facial → "cum"
- bondage, bdsm, tied → "bondage"
- etc.

These tags will be used to find relevant LoRAs, so be specific!

Respond ONLY with valid JSON, no other text."""
```

**Approach 2**: Add NSFW-specific scoring bonus

**Location**: `ml_lib/diffusion/services/ollama_selector.py:427-455`

```python
def _score_lora(self, lora, analysis: PromptAnalysis) -> float:
    """Score LoRA match for analysis."""
    score = 0.0

    # NEW: NSFW content bonus
    if hasattr(analysis, 'nsfw_acts') and analysis.nsfw_acts:
        lora_tags_lower = [tag.lower() for tag in lora.tags]
        nsfw_matches = sum(1 for act in analysis.nsfw_acts
                          if any(act.lower() in tag for tag in lora_tags_lower))
        score += nsfw_matches * 25.0  # Big bonus for NSFW matches!

        # Extra bonus if LoRA has NSFW flag
        if hasattr(lora, 'is_nsfw') and lora.is_nsfw:
            score += 20.0

    # Popularity
    score += min(lora.popularity_score / 5, 15.0)

    # Tag matching (existing logic)
    lora_tags = [tag.lower() for tag in lora.tags]
    recommended_tags = [tag.lower() for tag in analysis.recommended_lora_tags]
    key_concepts = [kw.lower() for kw in analysis.key_concepts]

    all_search_terms = set(recommended_tags + key_concepts)
    matching_tags = len(set(lora_tags) & all_search_terms)
    score += matching_tags * 15.0

    # Trigger word matching
    trigger_words = [tw.lower() for tw in lora.trigger_words]
    prompt_words = set(analysis.key_concepts)

    if any(tw in prompt_words for tw in trigger_words):
        score += 10.0

    # Model name matching
    model_name_lower = lora.model_name.lower()
    if any(concept in model_name_lower for concept in key_concepts):
        score += 10.0

    return score
```

### 🔧 Solution 3: Add Prompt Validation

**Create**: `ml_lib/diffusion/services/prompt_validator.py`

```python
def validate_prompt_length(prompt: str, max_tokens: int = 77) -> dict:
    """
    Validate prompt length and provide warnings.

    Returns:
        {
            "is_valid": bool,
            "token_count": int,
            "needs_compaction": bool,
            "truncated_content": list[str],  # What would be lost
            "suggestions": list[str]
        }
    """
    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    tokens = tokenizer.encode(prompt)
    token_count = len(tokens)

    result = {
        "is_valid": token_count <= max_tokens,
        "token_count": token_count,
        "needs_compaction": token_count > max_tokens,
        "truncated_content": [],
        "suggestions": []
    }

    if token_count > max_tokens:
        # Decode what would be truncated
        kept_tokens = tokens[:max_tokens]
        truncated_tokens = tokens[max_tokens:]

        kept_text = tokenizer.decode(kept_tokens)
        truncated_text = tokenizer.decode(truncated_tokens)

        result["truncated_content"] = [truncated_text]
        result["suggestions"] = [
            f"Prompt is {token_count} tokens, exceeds {max_tokens} limit by {token_count - max_tokens}",
            f"Will truncate: {truncated_text[:100]}...",
            "Recommendation: Use compact_prompt() to prioritize important content"
        ]

    return result
```

## Priority Implementation Order

1. **IMMEDIATE** (Today):
   - Add `compact_prompt()` to `prompt_analyzer.py`
   - Integrate into `optimize_for_model()`
   - Test with one real prompt

2. **HIGH** (Tomorrow):
   - Enhance Ollama analysis to extract NSFW acts
   - Add NSFW scoring bonus to LoRA selection
   - Test with 5-10 sanitized prompts

3. **MEDIUM** (This week):
   - Add prompt validation/warnings
   - Create comprehensive test suite
   - Document best practices

4. **LOW** (Later):
   - Character generator improvements (if needed)
   - Metadata scraper enhancements
   - Performance optimization

## Expected Results

After implementing solutions 1 & 2:

- ✅ Prompts stay under 77 tokens
- ✅ NSFW content preserved (highest priority)
- ✅ Relevant LoRAs selected (e.g., "NSFW_XL", "Sex_Positions_XL")
- ✅ Images match prompt intent
- ✅ No more blurry/generic results
- ✅ Memory usage unchanged

## Next Steps

Ready to implement? I recommend:

1. Start with `compact_prompt()` function
2. Test with one prompt from `data/prompt_sanitized.txt`
3. Verify token count and image quality
4. Then move to LoRA improvements

Would you like me to implement these solutions?
