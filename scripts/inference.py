# scripts/inference.py
"""
Inference script for single samples.

Deployment usage
----------------
Pass --hf_dir to the directory containing your saved HuggingFace files:
    tokenizer.json  tokenizer_config.json  spiece.model
    special_tokens_map.json  model.safetensors
    generation_config.json  config.json

These are used to reconstruct the tokenizer and provide the backbone
architecture template.  The actual trained weights come from --checkpoint.

If --hf_dir is omitted the script falls back to downloading from the Hub
(original behaviour, useful during development).
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import argparse
import sys
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent))

from src.models.model import MultimodalFakeNewsDetectorV2
from src.utils.logger import setup_logger
from transformers import AutoTokenizer

logger = setup_logger('inference', 'logs/inference.log')


class FakeNewsPredictor:
    """
    Predictor for single samples.

    Parameters
    ----------
    checkpoint_path : str
        Path to best_model.pt saved by the trainer.
    hf_dir : str | None
        Directory containing the saved HuggingFace files (tokenizer + backbone
        config/weights).  When provided the model runs fully offline.
        When None the original Hub ID stored in model_config is used.
    device : str
        'cuda' or 'cpu'.
    """

    def __init__(
        self,
        checkpoint_path: str,
        hf_dir: str = None,
        device: str = 'cuda',
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # ── Load checkpoint ───────────────────────────────────────────────
        logger.info(f'Loading checkpoint: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        model_config  = checkpoint['model_config']
        source_to_id  = checkpoint['source_to_id']

        # ── Resolve text-encoder source ───────────────────────────────────
        # If local HuggingFace files are provided, override the Hub ID so
        # that AutoTokenizer and the TextEncoder backbone both load offline.
        if hf_dir is not None:
            hf_dir = str(Path(hf_dir).resolve())
            logger.info(f'Using local HuggingFace files: {hf_dir}')
            model_config.text_encoder_name = hf_dir   # TextEncoder reads this
        else:
            logger.info(
                f'No --hf_dir supplied — loading from Hub: '
                f'{model_config.text_encoder_name}'
            )

        # ── Tokenizer ─────────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config.text_encoder_name
        )

        # ── Model ─────────────────────────────────────────────────────────
        # __init__ calls AutoModel.from_pretrained(model_config.text_encoder_name)
        # which now points at hf_dir (local) rather than the Hub.
        # load_state_dict then overwrites those weights with the trained ones.
        self.model = MultimodalFakeNewsDetectorV2(
            config               = model_config,
            tokenizer_vocab_size = len(self.tokenizer),
            source_to_id         = source_to_id,
        )
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.model.to(self.device)
        self.model.eval()

        # ── Image transform ───────────────────────────────────────────────
        self.image_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

        logger.info('Predictor initialized')
        logger.info(
            f'Parameters: {sum(p.numel() for p in self.model.parameters()):,}'
        )

    # ──────────────────────────────────────────────────────────────────────

    def preprocess_text(self, text: str) -> dict:
        encoding = self.tokenizer(
            text,
            max_length  = 512,
            padding     = 'max_length',
            truncation  = True,
            return_tensors = 'pt',
        )
        return {
            'input_ids':      encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device),
        }

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        try:
            image = np.array(Image.open(image_path).convert('RGB'))
        except Exception:
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        image = self.image_transform(image=image)['image']
        return image.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, text: str, image_path: str, source: str = None) -> dict:
        """
        Args:
            text:       Post text.
            image_path: Path to the post image.
            source:     Optional source name ('twitter', 'weibo', …).

        Returns:
            dict with prediction, confidence, probabilities, uncertainty,
            and Dirichlet evidence values.
        """
        text_inputs = self.preprocess_text(text)
        image       = self.preprocess_image(image_path)
        sources     = [source] if source else None

        outputs = self.model(
            input_ids      = text_inputs['input_ids'],
            attention_mask = text_inputs['attention_mask'],
            images         = image,
            sources        = sources,
        )

        probs      = outputs['classification']['prob'].cpu().numpy()[0]
        prediction = int(np.argmax(probs))

        result = {
            'prediction':       'FAKE' if prediction == 1 else 'REAL',
            'prediction_label': prediction,
            'confidence':       float(probs[prediction]),
            'prob_real':        float(probs[0]),
            'prob_fake':        float(probs[1]),
        }

        if 'uncertainty' in outputs['classification']:
            result['uncertainty'] = float(
                outputs['classification']['uncertainty'].cpu().item()
            )

        if 'alpha' in outputs['classification']:
            alpha = outputs['classification']['alpha'].cpu().numpy()[0]
            result['evidence'] = {
                'real': float(alpha[0]),
                'fake': float(alpha[1]),
            }

        return result


# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Run inference on a single sample',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to best_model.pt')
    parser.add_argument('--hf_dir', type=str, default=None,
                        help='Directory with saved HuggingFace files '
                             '(tokenizer.json, config.json, model.safetensors, …). '
                             'Enables fully offline inference. '
                             'Falls back to Hub download when omitted.')
    parser.add_argument('--text',   type=str, required=True, help='Post text')
    parser.add_argument('--image',  type=str, required=True, help='Path to image')
    parser.add_argument('--source', type=str, default=None,
                        help='Source name (twitter, weibo, …)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda / cpu)')

    args = parser.parse_args()

    if not Path(args.image).exists():
        logger.error(f'Image not found: {args.image}')
        print(f'ERROR: Image not found: {args.image}')
        return

    try:
        predictor = FakeNewsPredictor(
            checkpoint_path = args.checkpoint,
            hf_dir          = args.hf_dir,
            device          = args.device,
        )
    except Exception as e:
        logger.error(f'Failed to initialize predictor: {e}')
        print(f'ERROR: Failed to load model: {e}')
        return

    print('\n' + '='*80)
    print('Running inference…')
    print('='*80)

    try:
        result = predictor.predict(args.text, args.image, args.source)

        print('\n' + '='*80)
        print('PREDICTION RESULTS')
        print('='*80)
        print(f"Text:   {args.text}")
        print(f"Image:  {args.image}")
        if args.source:
            print(f"Source: {args.source}")
        print('-'*80)
        print(f"Prediction:       {result['prediction']}")
        print(f"Confidence:       {result['confidence']:.2%}")
        print('-'*80)
        print(f"Probability Real: {result['prob_real']:.2%}")
        print(f"Probability Fake: {result['prob_fake']:.2%}")

        if 'uncertainty' in result:
            u = result['uncertainty']
            interp = (
                'High confidence'             if u < 0.3 else
                'Moderate confidence'         if u < 0.6 else
                'Low confidence (uncertain)'
            )
            print(f"Uncertainty:      {u:.4f}  ({interp})")

        if 'evidence' in result:
            print('-'*80)
            print('Evidence (Dirichlet α):')
            print(f"  Real: {result['evidence']['real']:.2f}")
            print(f"  Fake: {result['evidence']['fake']:.2f}")

        print('='*80 + '\n')

        logger.info(f"Prediction: {result['prediction']}  "
                    f"confidence={result['confidence']:.4f}  "
                    f"uncertainty={result.get('uncertainty', 'n/a')}")

    except Exception as e:
        logger.error(f'Prediction failed: {e}')
        print(f'\nERROR: Prediction failed: {e}')
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()