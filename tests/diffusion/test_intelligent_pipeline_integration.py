"""Integration test for intelligent generation pipeline."""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock
from PIL import Image

# Pipeline components
from ml_lib.diffusion.intelligent.pipeline.services import (
    IntelligentGenerationPipeline,
    BatchProcessor,
    DecisionExplainer,
    FeedbackCollector,
)
from ml_lib.diffusion.intelligent.pipeline.entities import (
    PipelineConfig,
    OperationMode,
    GenerationConstraints,
    LoRAPreferences,
    MemorySettings,
    Priority,
    BatchConfig,
    VariationStrategy,
)


class TestIntelligentPipelineIntegration:
    """Integration tests for the complete intelligent pipeline."""

    @pytest.fixture
    def mock_subsystems(self):
        """Create mock subsystems for testing."""
        # Mock analysis result
        mock_analysis = Mock()
        mock_analysis.complexity_category = Mock(value="moderate")
        mock_analysis.identified_concepts = ["anime", "girl", "magical"]
        mock_analysis.intent = Mock(artistic_style=Mock(value="anime"))

        # Mock LoRA recommendations
        mock_lora_rec = Mock()
        mock_lora_rec.lora_name = "anime_style_v2"
        mock_lora_rec.suggested_alpha = 0.8
        mock_lora_rec.confidence_score = 0.85
        mock_lora_rec.reasoning = "Matched 'anime' keyword"

        # Mock optimized parameters
        mock_params = Mock()
        mock_params.num_steps = 35
        mock_params.guidance_scale = 7.5
        mock_params.width = 1024
        mock_params.height = 1024
        mock_params.sampler_name = "DPM++ 2M Karras"
        mock_params.estimated_vram_gb = 8.5
        mock_params.estimated_time_seconds = 45.0

        return {
            "analysis": mock_analysis,
            "lora_recs": [mock_lora_rec],
            "params": mock_params,
        }

    @pytest.fixture
    def pipeline_config(self):
        """Create test pipeline configuration."""
        return PipelineConfig(
            base_model="stabilityai/sdxl-base-1.0",
            mode=OperationMode.AUTO,
            constraints=GenerationConstraints(
                max_time_seconds=120,
                max_vram_gb=12.0,
                priority=Priority.QUALITY,
            ),
            lora_preferences=LoRAPreferences(
                max_loras=3,
                min_confidence=0.6,
                allow_style_mixing=True,
            ),
            memory_settings=MemorySettings(
                max_vram_gb=12.0,
                offload_strategy="balanced",
            ),
            enable_learning=True,
        )

    def test_pipeline_initialization(self, pipeline_config):
        """Test pipeline initializes correctly."""
        pipeline = IntelligentGenerationPipeline(config=pipeline_config)

        assert pipeline.config.mode == OperationMode.AUTO
        assert pipeline.config.constraints.priority == Priority.QUALITY
        assert pipeline.diffusion_pipeline is None  # Not loaded yet

    def test_simple_generation_workflow(self, pipeline_config, mock_subsystems, monkeypatch):
        """
        Test end-to-end generation workflow.

        Workflow:
        1. Create pipeline
        2. Call generate() with prompt
        3. Verify result structure
        """
        # Create pipeline
        pipeline = IntelligentGenerationPipeline(config=pipeline_config)

        # Mock subsystem methods
        if hasattr(pipeline, "prompt_analyzer") and pipeline.prompt_analyzer:
            pipeline.prompt_analyzer.analyze = Mock(return_value=mock_subsystems["analysis"])
        if hasattr(pipeline, "lora_recommender") and pipeline.lora_recommender:
            pipeline.lora_recommender.recommend = Mock(return_value=mock_subsystems["lora_recs"])
        if hasattr(pipeline, "param_optimizer") and pipeline.param_optimizer:
            pipeline.param_optimizer.optimize = Mock(return_value=mock_subsystems["params"])

        # Mock image generation
        def mock_generate_image(*args, **kwargs):
            return Image.new("RGB", (1024, 1024), color="gray")

        pipeline._generate_image = mock_generate_image

        # Generate
        result = pipeline.generate(
            prompt="anime girl with magical powers",
            negative_prompt="low quality",
            seed=42,
        )

        # Verify result
        assert result is not None
        assert result.image is not None
        assert result.metadata is not None
        assert result.explanation is not None
        assert result.metadata.seed == 42
        assert result.metadata.prompt == "anime girl with magical powers"

    def test_assisted_mode_workflow(self, pipeline_config, mock_subsystems):
        """
        Test ASSISTED mode workflow.

        Workflow:
        1. Get recommendations
        2. User reviews/modifies
        3. Generate from modified recommendations
        """
        # Create pipeline in ASSISTED mode
        pipeline_config.mode = OperationMode.ASSISTED
        pipeline = IntelligentGenerationPipeline(config=pipeline_config)

        # Mock subsystems
        if hasattr(pipeline, "prompt_analyzer") and pipeline.prompt_analyzer:
            pipeline.prompt_analyzer.analyze = Mock(return_value=mock_subsystems["analysis"])
        if hasattr(pipeline, "lora_recommender") and pipeline.lora_recommender:
            pipeline.lora_recommender.recommend = Mock(return_value=mock_subsystems["lora_recs"])
        if hasattr(pipeline, "param_optimizer") and pipeline.param_optimizer:
            pipeline.param_optimizer.optimize = Mock(return_value=mock_subsystems["params"])

        # Get recommendations
        recommendations = pipeline.analyze_and_recommend("anime girl")

        assert recommendations is not None
        assert recommendations.suggested_loras is not None
        assert recommendations.suggested_params is not None

        # User modifies parameters
        recommendations.suggested_params.num_steps = 50

        # Mock generation
        pipeline._generate_image = Mock(return_value=Image.new("RGB", (1024, 1024)))

        # Generate from modified recommendations
        result = pipeline.generate_from_recommendations(
            prompt="anime girl",
            recommendations=recommendations,
            seed=42,
        )

        assert result is not None
        assert result.metadata.steps == 50  # Modified value

    def test_batch_generation(self, pipeline_config, mock_subsystems):
        """Test batch generation with seed variation."""
        pipeline = IntelligentGenerationPipeline(config=pipeline_config)

        # Mock subsystems
        if hasattr(pipeline, "prompt_analyzer") and pipeline.prompt_analyzer:
            pipeline.prompt_analyzer.analyze = Mock(return_value=mock_subsystems["analysis"])
        if hasattr(pipeline, "lora_recommender") and pipeline.lora_recommender:
            pipeline.lora_recommender.recommend = Mock(return_value=mock_subsystems["lora_recs"])
        if hasattr(pipeline, "param_optimizer") and pipeline.param_optimizer:
            pipeline.param_optimizer.optimize = Mock(return_value=mock_subsystems["params"])

        pipeline._generate_image = Mock(return_value=Image.new("RGB", (1024, 1024)))

        # Create batch processor
        processor = BatchProcessor(pipeline)

        batch_config = BatchConfig(
            num_images=3,
            variation_strategy=VariationStrategy.SEED_VARIATION,
            base_seed=100,
        )

        # Generate batch
        results = processor.process_batch(
            prompt="anime girl",
            config=batch_config,
        )

        # Verify
        assert len(results) == 3
        assert all(r.image is not None for r in results)
        # Verify different seeds
        seeds = [r.metadata.seed for r in results]
        assert len(set(seeds)) == 3  # All unique

    def test_feedback_collection(self, pipeline_config):
        """Test feedback collection and learning integration."""
        from ml_lib.diffusion.intelligent.pipeline.services.feedback_collector import (
            UserFeedback,
        )
        from ml_lib.diffusion.intelligent.prompting.services.learning_engine import (
            LearningEngine,
        )
        from datetime import datetime
        import tempfile

        # Create learning engine with temp DB
        with tempfile.TemporaryDirectory() as tmpdir:
            learning_engine = LearningEngine(db_path=Path(tmpdir) / "test.db")

            # Create feedback collector
            collector = FeedbackCollector(learning_engine=learning_engine)

            # Mock recommendations
            mock_recommendations = Mock()
            mock_recommendations.suggested_loras = [
                Mock(lora_name="anime_style"),
                Mock(lora_name="detail_enhancer"),
            ]
            mock_recommendations.suggested_params = Mock()
            mock_recommendations.suggested_params.num_steps = 35
            mock_recommendations.suggested_params.guidance_scale = 7.5
            mock_recommendations.suggested_params.width = 1024
            mock_recommendations.suggested_params.height = 1024
            mock_recommendations.suggested_params.sampler_name = "DPM++ 2M"

            # Start session
            session = collector.start_session(
                generation_id="test_001",
                prompt="anime girl",
                negative_prompt="",
                recommendations=mock_recommendations,
            )

            assert session.generation_id == "test_001"

            # Collect feedback
            feedback = UserFeedback(
                generation_id="test_001",
                timestamp=datetime.now().isoformat(),
                rating=5,
                comments="Perfect!",
                tags=["anime", "character"],
                saved=True,
            )

            collector.collect_feedback(feedback, session)

            # Verify feedback was recorded
            assert len(collector.feedback_history) == 1

            # Verify learning engine received feedback
            insights = learning_engine.get_insights()
            assert insights["total_feedback_records"] == 1
            assert insights["overall_average_rating"] == 5.0

    def test_decision_explainer(self, mock_subsystems):
        """Test decision explainer functionality."""
        explainer = DecisionExplainer()

        # Test LoRA explanation
        explanation = explainer.explain_lora_selection(
            lora_name="anime_style_v2",
            confidence=0.85,
            reasoning="Matched 'anime' keyword with high semantic similarity",
            alternative_loras=[("realistic_style", 0.45), ("painterly", 0.40)],
        )

        assert "anime_style_v2" in explanation
        assert "0.85" in explanation

        # Test parameter explanation
        param_explanation = explainer.explain_parameter_choice(
            param_name="num_steps",
            param_value=40,
            reasoning="High complexity prompt requires more steps",
            default_value=30,
        )

        assert "40" in param_explanation
        assert "complexity" in param_explanation.lower()

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = PipelineConfig(
            base_model="sdxl-base-1.0",
            constraints=GenerationConstraints(
                max_time_seconds=60,
                max_vram_gb=8.0,
                priority=Priority.QUALITY,
            ),
        )
        assert config.base_model == "sdxl-base-1.0"

        # Invalid constraints
        with pytest.raises(ValueError):
            GenerationConstraints(max_time_seconds=-10)

        with pytest.raises(ValueError):
            LoRAPreferences(min_confidence=1.5)

    def test_memory_settings_validation(self):
        """Test memory settings validation."""
        # Valid settings
        settings = MemorySettings(
            max_vram_gb=12.0,
            offload_strategy="balanced",
            enable_quantization=True,
            quantization_dtype="fp16",
        )
        assert settings.max_vram_gb == 12.0

        # Invalid offload strategy
        with pytest.raises(ValueError):
            MemorySettings(offload_strategy="invalid")

        # Invalid quantization dtype
        with pytest.raises(ValueError):
            MemorySettings(quantization_dtype="fp32")

    def test_batch_config_validation(self):
        """Test batch config validation."""
        # Valid config
        config = BatchConfig(
            num_images=4,
            variation_strategy=VariationStrategy.SEED_VARIATION,
            base_seed=42,
        )
        assert config.num_images == 4

        # Invalid num_images
        with pytest.raises(ValueError):
            BatchConfig(num_images=0)

        # Invalid base_seed
        with pytest.raises(ValueError):
            BatchConfig(base_seed=-1)

    def test_generation_result_saving(self, tmp_path):
        """Test saving generation results with metadata."""
        from ml_lib.diffusion.intelligent.pipeline.entities import (
            GenerationResult,
            GenerationMetadata,
            GenerationExplanation,
            LoRAInfo,
        )

        # Create test result
        metadata = GenerationMetadata(
            prompt="test prompt",
            negative_prompt="",
            seed=42,
            steps=35,
            cfg_scale=7.5,
            width=1024,
            height=1024,
            sampler="DPM++ 2M",
            loras_used=[LoRAInfo(name="test_lora", alpha=0.8)],
            generation_time_seconds=45.0,
            peak_vram_gb=8.5,
            base_model_id="sdxl-base-1.0",
        )

        explanation = GenerationExplanation(
            summary="Test generation",
            lora_reasoning={"test_lora": "Test reasoning"},
            parameter_reasoning={"steps": "35 steps for quality"},
            performance_notes=["Generated in 45s"],
        )

        result = GenerationResult(
            id="test_001",
            image=Image.new("RGB", (1024, 1024)),
            metadata=metadata,
            explanation=explanation,
        )

        # Save
        output_path = tmp_path / "test_output.png"
        result.save(output_path, save_metadata=True, save_explanation=True)

        # Verify files exist
        assert output_path.exists()
        assert (tmp_path / "test_output.explanation.json").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
