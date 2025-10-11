"""Batch processor for generating multiple images with variations."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Callable, Any
import random

from ..entities import (
    BatchConfig,
    GenerationResult,
    VariationStrategy,
)

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Batch processor for generating multiple images with variations.

    Supports different variation strategies:
    - SEED_VARIATION: Same params, different seeds
    - PARAM_VARIATION: Vary steps, CFG, etc.
    - LORA_VARIATION: Try different LoRA combinations
    - MIXED: Combine multiple strategies

    Example:
        >>> processor = BatchProcessor(pipeline)
        >>> config = BatchConfig(
        ...     num_images=4,
        ...     variation_strategy=VariationStrategy.SEED_VARIATION,
        ...     output_dir="./batch_output"
        ... )
        >>> results = processor.process_batch(
        ...     "anime girl with magical powers",
        ...     config
        ... )
    """

    def __init__(self, pipeline: Any):
        """
        Initialize batch processor.

        Args:
            pipeline: IntelligentGenerationPipeline instance
        """
        self.pipeline = pipeline

    def process_batch(
        self,
        prompt: str,
        config: BatchConfig,
        negative_prompt: str = "",
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list[GenerationResult]:
        """
        Process batch generation with variations.

        Args:
            prompt: Text prompt
            config: Batch configuration
            negative_prompt: Negative prompt (optional)
            progress_callback: Optional callback (current, total) for progress

        Returns:
            List of GenerationResults

        Example:
            >>> def progress(current, total):
            ...     print(f"Progress: {current}/{total}")
            >>> results = processor.process_batch(
            ...     "anime girl",
            ...     config,
            ...     progress_callback=progress
            ... )
        """
        logger.info(
            f"Starting batch generation: {config.num_images} images "
            f"with {config.variation_strategy.value} strategy"
        )

        # Choose strategy handler
        if config.variation_strategy == VariationStrategy.SEED_VARIATION:
            results = self._seed_variation(
                prompt, config, negative_prompt, progress_callback
            )
        elif config.variation_strategy == VariationStrategy.PARAM_VARIATION:
            results = self._param_variation(
                prompt, config, negative_prompt, progress_callback
            )
        elif config.variation_strategy == VariationStrategy.LORA_VARIATION:
            results = self._lora_variation(
                prompt, config, negative_prompt, progress_callback
            )
        elif config.variation_strategy == VariationStrategy.MIXED:
            results = self._mixed_variation(
                prompt, config, negative_prompt, progress_callback
            )
        else:
            raise ValueError(f"Unknown variation strategy: {config.variation_strategy}")

        # Save if output directory specified
        if config.output_dir and config.save_individually:
            self._save_batch_results(results, config.output_dir)

        logger.info(f"Batch generation completed: {len(results)} images")
        return results

    def _seed_variation(
        self,
        prompt: str,
        config: BatchConfig,
        negative_prompt: str,
        progress_callback: Optional[Callable[[int, int], None]],
    ) -> list[GenerationResult]:
        """Generate with same params but different seeds."""
        # Analyze once and reuse
        recommendations = self.pipeline.analyze_and_recommend(prompt)

        results = []
        base_seed = config.base_seed if config.base_seed is not None else random.randint(0, 10000)

        for i in range(config.num_images):
            seed = base_seed + i

            logger.debug(f"Generating image {i+1}/{config.num_images} (seed={seed})")

            result = self.pipeline.generate_from_recommendations(
                prompt=prompt,
                recommendations=recommendations,
                negative_prompt=negative_prompt,
                seed=seed,
            )

            results.append(result)

            if progress_callback:
                progress_callback(i + 1, config.num_images)

        return results

    def _param_variation(
        self,
        prompt: str,
        config: BatchConfig,
        negative_prompt: str,
        progress_callback: Optional[Callable[[int, int], None]],
    ) -> list[GenerationResult]:
        """Generate with parameter variations."""
        # Get base recommendations
        base_recs = self.pipeline.analyze_and_recommend(prompt)

        # Generate parameter variations
        param_variations = self._generate_param_variations(
            base_recs, config.num_images
        )

        results = []
        for i, params in enumerate(param_variations):
            logger.debug(
                f"Generating image {i+1}/{config.num_images} "
                f"(steps={params['num_steps']}, cfg={params['guidance_scale']})"
            )

            result = self.pipeline.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                **params,
            )

            results.append(result)

            if progress_callback:
                progress_callback(i + 1, config.num_images)

        return results

    def _lora_variation(
        self,
        prompt: str,
        config: BatchConfig,
        negative_prompt: str,
        progress_callback: Optional[Callable[[int, int], None]],
    ) -> list[GenerationResult]:
        """Generate with different LoRA combinations."""
        # Note: This is a simplified version
        # In production, we'd actually vary which LoRAs are used

        results = []
        for i in range(config.num_images):
            logger.debug(f"Generating image {i+1}/{config.num_images} (LoRA variation)")

            # For now, just use different seeds
            # In production, modify LoRA selection
            result = self.pipeline.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=i if config.base_seed is None else config.base_seed + i,
            )

            results.append(result)

            if progress_callback:
                progress_callback(i + 1, config.num_images)

        return results

    def _mixed_variation(
        self,
        prompt: str,
        config: BatchConfig,
        negative_prompt: str,
        progress_callback: Optional[Callable[[int, int], None]],
    ) -> list[GenerationResult]:
        """Generate with mixed variations (seeds + params)."""
        results = []

        # Mix: alternate between seed variation and param variation
        base_recs = self.pipeline.analyze_and_recommend(prompt)
        param_variations = self._generate_param_variations(
            base_recs, config.num_images // 2
        )

        for i in range(config.num_images):
            if i % 2 == 0 and param_variations:
                # Use param variation
                params = param_variations[i // 2] if i // 2 < len(param_variations) else {}
                result = self.pipeline.generate(
                    prompt=prompt, negative_prompt=negative_prompt, **params
                )
            else:
                # Use seed variation
                seed = (config.base_seed or 0) + i
                result = self.pipeline.generate_from_recommendations(
                    prompt=prompt,
                    recommendations=base_recs,
                    negative_prompt=negative_prompt,
                    seed=seed,
                )

            results.append(result)

            if progress_callback:
                progress_callback(i + 1, config.num_images)

        return results

    def _generate_param_variations(
        self, base_recs: Any, num_variations: int
    ) -> list[dict[str, Any]]:
        """
        Generate parameter variations from base recommendations.

        Args:
            base_recs: Base recommendations
            num_variations: Number of variations to generate

        Returns:
            List of parameter dictionaries
        """
        base_params = base_recs.suggested_params
        variations = []

        # Vary steps
        step_values = [
            base_params.num_steps - 10,
            base_params.num_steps,
            base_params.num_steps + 10,
        ]

        # Vary CFG scale
        cfg_values = [
            base_params.guidance_scale - 1.0,
            base_params.guidance_scale,
            base_params.guidance_scale + 1.0,
        ]

        for i in range(num_variations):
            variation = {
                "num_steps": step_values[i % len(step_values)],
                "guidance_scale": cfg_values[i % len(cfg_values)],
            }
            variations.append(variation)

        return variations

    def _save_batch_results(self, results: list[GenerationResult], output_dir: str):
        """
        Save batch results to directory.

        Args:
            results: List of generation results
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for i, result in enumerate(results):
            # Save image
            image_path = output_path / f"image_{i:04d}.png"
            result.save(
                image_path,
                save_metadata=True,
                save_explanation=True,
            )

            logger.debug(f"Saved image {i+1} to {image_path}")

        logger.info(f"Saved {len(results)} images to {output_dir}")

    def process_batch_parallel(
        self,
        prompt: str,
        config: BatchConfig,
        negative_prompt: str = "",
        max_workers: int = 2,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list[GenerationResult]:
        """
        Process batch generation in parallel.

        WARNING: Requires significant VRAM for multiple concurrent generations.

        Args:
            prompt: Text prompt
            config: Batch configuration
            negative_prompt: Negative prompt (optional)
            max_workers: Maximum parallel workers
            progress_callback: Optional progress callback

        Returns:
            List of GenerationResults
        """
        logger.info(
            f"Starting parallel batch generation: {config.num_images} images "
            f"with {max_workers} workers"
        )
        logger.warning("Parallel generation requires significant VRAM!")

        # Only seed variation supported for parallel
        if config.variation_strategy != VariationStrategy.SEED_VARIATION:
            logger.warning(
                "Only SEED_VARIATION supported for parallel - falling back to sequential"
            )
            return self.process_batch(prompt, config, negative_prompt, progress_callback)

        # Analyze once
        recommendations = self.pipeline.analyze_and_recommend(prompt)

        # Generate seeds
        base_seed = config.base_seed if config.base_seed is not None else random.randint(0, 10000)
        seeds = [base_seed + i for i in range(config.num_images)]

        # Parallel generation
        results = []
        completed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(
                    self.pipeline.generate_from_recommendations,
                    prompt,
                    recommendations,
                    negative_prompt,
                    seed,
                ): seed
                for seed in seeds
            }

            # Collect results as they complete
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                completed += 1

                if progress_callback:
                    progress_callback(completed, config.num_images)

                logger.debug(f"Completed {completed}/{config.num_images}")

        logger.info(f"Parallel batch generation completed: {len(results)} images")
        return results
