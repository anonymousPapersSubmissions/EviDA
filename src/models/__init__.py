"""
Models package
"""

from .model import MultimodalFakeNewsDetectorV2
from .encoders import TextEncoder, VisionEncoder
from .fusion import MultimodalFusion
from .classifier import (
    EvidentialClassifierV2,
    StandardClassifier,
    ExplanationGenerator
)
from .domain_adaptation import (
    GradientReversalLayer,
    DomainDiscriminator,
    DomainSpecificBatchNorm,
    LambdaScheduler
)

__all__ = [
    'MultimodalFakeNewsDetectorV2',
    'TextEncoder',
    'VisionEncoder',
    'MultimodalFusion',
    'EvidentialClassifierV2',
    'StandardClassifier',
    'ExplanationGenerator',
    'GradientReversalLayer',
    'DomainDiscriminator',
    'DomainSpecificBatchNorm',
    'LambdaScheduler',
]