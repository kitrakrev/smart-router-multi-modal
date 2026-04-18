#!/usr/bin/env python3
"""
Train routing-quality sentence embeddings via contrastive learning.

Uses MultipleNegativesRankingLoss on LMSYS 55K conversation data to learn
embeddings that cluster queries by task type rather than surface similarity.

The trained model is pushed to HuggingFace: kitrakrev/smart-router-embeddings

Usage:
  python train_routing_embeddings.py                 # train
  python train_routing_embeddings.py --push-to-hub   # train + push
"""

# TODO: Implementation — see finetune_lmsys.py for the multi-head approach.
#       This script focuses on the embedding layer only (no classification heads).
raise NotImplementedError(
    "Contrastive embedding training is currently handled by finetune_lmsys.py. "
    "This file is a placeholder for a standalone embedding-only trainer."
)
