from torchvision import transforms
from scripts.training.utils import get_train_transform, get_test_transform


class TestTrainingTransforms:
    """Tests for task-specific data augmentations."""

    def test_species_transform_contains_color_jitter(self):
        """Species transform should include ColorJitter."""
        transform = get_train_transform(task="species")
        transform_types = [type(t) for t in transform.transforms]
        assert transforms.ColorJitter in transform_types, (
            "ColorJitter should be present in species training transform"
        )

    def test_freshness_transform_omits_color_jitter(self):
        """Freshness transform should omit ColorJitter to keep color cues."""
        transform = get_train_transform(task="freshness")
        transform_types = [type(t) for t in transform.transforms]
        assert transforms.ColorJitter not in transform_types, (
            "ColorJitter must not be present in freshness transform"
        )

    def test_test_transform_structure(self):
        """Test transform should resize to 224x224 and normalize."""
        transform = get_test_transform()
        transform_types = [type(t) for t in transform.transforms]
        assert transforms.Resize in transform_types
        assert transforms.ToTensor in transform_types
        assert transforms.Normalize in transform_types
