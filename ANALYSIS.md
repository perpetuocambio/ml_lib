# Analysis of Current Issues & Action Plan

## Issues Identified

### 1. Prompt Truncation (CRITICAL)
**Location**: Intelligent Builder generation
**Problem**: CLIP model only handles 77 tokens, but prompts are 251 tokens
**Impact**: All explicit content is being cut off during generation
**Evidence**: Lines 60-62 in test_nsfw_fix.log show truncation warning

```
Token indices sequence length is longer than the specified maximum sequence length for this model (251 > 77)
The following part of your input was truncated...
```

**Result**: Images are blurry/not NSFW because the actual NSFW content is being removed!

### 2. Poor LoRA Selection
**Location**: `ml_lib/diffusion/services/ollama_selector.py:342-395`
**Problem**: LoRA scoring is based on tag matching but isn't finding NSFW-specific LoRAs
**Evidence**: Selected LoRAs for NSFW prompt: `Clothes_Pull_XL-000010`, `MS_DAH_SDXL_V1`, `clnmdmccXLrd`
**Expected**: Should select LoRAs tagged with "nsfw", "explicit", "sex", etc.

### 3. Character Generator Not Suitable for Production
**Location**: `ml_lib/diffusion/models/character.py`
**Problem**: Generates single-character portraits, but real users want:
- Multi-character scenes (threesome, group, etc.)
- Complex poses (fellatio, missionary, doggystyle, etc.)
- Specific scenarios (caught, phone pov, classroom, etc.)
- Detailed expressions (ahegao, blushing, moaning, etc.)

**Analysis of Real Prompts**:
- 90% include multiple characters (1girl + 1-2boys minimum)
- 85% specify explicit sexual acts
- 70% include specific scenarios/settings
- 60% include detailed expressions and body language

### 4. Prompt Analysis Quality
**Location**: `ml_lib/diffusion/services/ollama_selector.py:147-286`
**Problem**: Ollama analysis is basic, doesn't extract NSFW-specific tags
**Current**: Extracts "realistic", "portrait", "character"
**Needed**: Should extract "fellatio", "anal", "bukkake", "cum", etc. for LoRA matching

## Solutions

### Solution 1: Fix Prompt Truncation
**Approach**: Implement prompt compaction/prioritization

```python
def compact_prompt(prompt: str, max_tokens: int = 77) -> str:
    """
    Compact long prompts intelligently:
    1. Remove quality tags (masterpiece, best quality, etc.)
    2. Remove redundant tags
    3. Prioritize NSFW content over quality markers
    4. Use CLIP tokenizer to measure actual token count
    """
    pass
```

**Alternative**: Use Long-CLIP or SDXL's longer context (works with SDXL!)

### Solution 2: Improve LoRA Selection
**Approach 1**: Add NSFW-specific scoring

```python
def _score_lora_nsfw(self, lora, analysis: PromptAnalysis) -> float:
    """
    Additional scoring for NSFW content:
    - Bonus for tags: nsfw, explicit, sex, fellatio, etc.
    - Bonus if lora.is_nsfw flag is set
    - Match explicit acts from prompt analysis
    """
    pass
```

**Approach 2**: Enhance Ollama analysis to extract explicit acts

```python
"recommended_lora_tags": [
    "nsfw", "explicit", "fellatio", "cum", "sex",
    "blonde", "big_breasts", "pov"
]
```

### Solution 3: Enhance Character Generator (Optional)
**Note**: This is not urgent since users provide their own prompts

Options:
1. Add multi-character support
2. Add explicit pose templates
3. Add scenario generation

**Recommendation**: Skip for now, focus on fixing the pipeline issues

## Priority Actions

### Task 1: Fix Prompt Truncation ⚡ URGENT
**File**: `ml_lib/diffusion/services/intelligent_builder.py`
**Changes**:
1. Add prompt compaction before generation
2. Prioritize explicit content over quality tags
3. Log warnings when compaction happens

### Task 2: Improve LoRA Selection ⚡ HIGH
**Files**:
- `ml_lib/diffusion/services/ollama_selector.py` (analysis)
- `ml_lib/diffusion/services/ollama_selector.py` (scoring)

**Changes**:
1. Enhance Ollama prompt to extract NSFW tags
2. Add NSFW-specific scoring bonuses
3. Filter out irrelevant LoRAs

### Task 3: Test with Real Prompts
**File**: Create `tests/test_production_prompts.py`
**Actions**:
1. Use sanitized prompts from data/prompt_sanitized.txt
2. Test model selection quality
3. Verify image quality improvements
4. Measure VRAM usage

## Expected Results

After fixes:
- ✅ Prompts under 77 tokens or properly compacted
- ✅ Relevant NSFW LoRAs selected (e.g., "NSFW_XL", "Sex_Positions_XL")
- ✅ Images match prompt intent (not blurry, actually NSFW)
- ✅ No memory issues

## Implementation Order

1. **First**: Fix prompt truncation (causes blurry images)
2. **Second**: Improve LoRA selection (ensures quality)
3. **Third**: Test with real prompts (validates fixes)
4. **Last**: Document and optimize
