# EVA-X-S CheXpert preparation (2026-11-08)

This is engineering preparation for a possible later fixed experiment. No EVA-X training, real-data inference, AUC evaluation, or competition submission has occurred. It does not replace the current CXR-LT incumbent or its test CSV.

## Reproducible source

- Model: `MapleF/eva_x`, revision `35ddcd6dab6ca99bbdb6cb45c8d1b093aefbd0ee`.
- Checkpoint: `eva_x_small_patch16_merged520k_mim_chexpert_ft.pth`, 260,339,517 bytes, SHA256 `1a5b63570ca4094c1118018e44d3012a95b904132942a8edb2fc0e5dc8aa0e97`.
- Author code: `hustvl/EVA-X`, commit `eedd642c85339b25682ceb574b644cb9afe88e62`.
- Local directory: `weights/medical/eva-x-small-chexpert`. `provenance.json` records pinned URLs, sizes and hashes for all downloaded files. The upstream Apache-2.0 license and the model card's MIT metadata are preserved.
- Download recipe: `outputs/eva-x-preparation-20261108/fetch_pinned.py`. Existing downloaded artifacts are reused only after the expected checkpoint size/hash pass.

## Corrections established from the actual checkpoint

The checkpoint's own `args.model` and the pinned CheXpert launch script both identify `eva02_small_patch16_xattn_fusedLN_SwiGLU_preln_RoPE`. It contains rotary buffers, separate query/value biases and packed SwiGLU weights. The upstream `models_vit.py` is a different vanilla ViT architecture and must not load this checkpoint.

The author-supported root `eva_x.py` provides a timm EVA implementation. Construct its `EVA_X` with image size224, patch16, embedding384, depth12, heads6, MLP ratio `4*2/3`, SwiGLU, rotary positions with reference14x14, global average pooling and five output classes. LayerNorm epsilon is1e-6. The author's checkpoint filter maps packed `mlp.w12`/`mlp.w3` to `mlp.fc1`/`mlp.fc2`. Strict learned-state loading succeeds; no missing learned parameters are accepted.

The source checkpoint also stores26 fixed rotary buffers: root sine/cosine and their identical copies in12 attention blocks. Each is196x64. Regenerated float32 timm buffers differ by up to0.00024330616; rounding both versions to float16 makes them exactly equal. This is consistent with a prior half-precision round trip, but its history is not established. To reproduce the stored model, verify all26 original buffers and restore the exact root sine/cosine values into the static196x128 concatenated `[sin, cos]` buffer. Do not silently replace them with freshly generated values.

Restricted `torch.load(weights_only=True,map_location="cpu")` remains enabled. This official checkpoint needs only `argparse.Namespace` allowlisted for legacy metadata. No unrestricted pickle loader is used. The checkpoint contains190 model tensors; its five-class head is5x384, and all tensors were checked finite.

## Feature contract and checks

Use RGB, shorter-edge256 bicubic resize, center crop224, `ToTensor`, then per-channel mean0.49185243/std0.28509309, as specified in the pinned evaluation transform. The local processor must match this recipe exactly on synthetic grayscale and RGB images.

The author timm model returns patch/class tokens from `forward_features`. Use `forward_head(tokens,pre_logits=True)` for the384-dimensional normalized patch mean, then the original linear head for five raw logits. Concatenate these into389D. Keep all model parameters frozen and all modules in eval mode. The five pretrained outputs are Atelectasis, Cardiomegaly, Consolidation, Edema and Pleural Effusion; a complete local ten-class head still needs a separately fixed training experiment.

Engineering evidence is under `outputs/eva-x-preparation-20261108`: `weight_inspection.json` records the restricted load and shape checks; `author_architecture_check.json` records strict author-model loading, the rotary discrepancy and exact restoration, and a finite synthetic384+5 CPU forward. The final `check_cpu.py` check passed on the actual checkpoint using two synthetic grayscale/RGB images: all learned mapped tensors exactly match, all26 fixed rotary buffers exactly match, preprocessing matches the official recipe, and389D feature replay is exact with maximum error0. Results are in `cpu_engineering_check.json`. All model parameters remain frozen and all modules in eval mode. The7 focused synthetic unittest cases and source compilation also passed. These checks use no outcome labels and do not establish a performance benefit.

Before any future fitting, freeze the feature definition, sources, preprocessing and patient-grouped comparison plan. Preserve the currently running stage-finetuning experiment and its frozen inputs. Public-pretraining overlap with the anonymized local cohort remains unknown.
