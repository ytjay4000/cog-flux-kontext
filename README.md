# FLUX.1 Kontext (Cog inference)

Compact Cog wrapper around Black Forest Labs' FLUX.1 *Kontext* dev model.  It loads the
Transformer, auto-encoder, CLIP/T5 text encoders, and optional NSFW safety checker,
then exposes a single `predict` endpoint that performs image-to-image editing or
style transfer conditioned on a text prompt.

```bash
# basic usage
cog predict -i prompt="make the hair blue" -i input_image=@lady.png
```

Everything required (weights download, Torch 2 compilation, etc.) happens
automatically on first run.

Licensed under Apache-2.0 for the wrapper code; see model card for FLUX.1 license.

### Performance Optimizations
- `torch.compile` is used in dynamic mode
- the two linear layers in the single stream block are quantized to run in FP8, using a modified version of aredden's [fp8 linear layer](https://github.com/aredden/flux-fp8-api/blob/main/float8_quantize.py)
- [taylor seer](https://arxiv.org/abs/2503.06923) style activation caching, enabled by the `go_fast` option in the cog predictor. May cause quality degradation for more complex editing tasks.
- enable pytorch's cudnn attention backend
