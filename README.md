# cog-flux-kontext

This is a [Cog](https://cog.run) inference model for FLUX.1 kontext [dev] by [Black Forest Labs](https://blackforestlabs.ai/). It powers the following Replicate model:

* https://replicate.com/black-forest-labs/flux-kontext-dev

## Features

* Compilation with `torch.compile`
* NSFW checking with [CompVis](https://huggingface.co/CompVis/stable-diffusion-safety-checker) and [Falcons.ai](https://huggingface.co/Falconsai/nsfw_image_detection) safety checkers

## Getting started

If you just want to use the models, you can run [FLUX.1 kontext [dev]](https://replicate.com/black-forest-labs/flux-kontext-dev) on Replicate with an API or in the browser.

The code in this repo can be used as a template for customizations on FLUX.1 kontext [dev], or to run the models on your own hardware.

You can run a single prediction on the model using:

```shell
cog predict -i prompt="make the hair green" -i input_image=@lady.png
```

The [Cog getting started guide](https://cog.run/getting-started/) explains what Cog is and how it works.

To deploy it to Replicate, run:

```shell
cog login
cog push r8.im/<your-username>/<your-model-name>
```

Learn more on [the deploy a custom model guide in the Replicate documentation](https://replicate.com/docs/guides/deploy-a-custom-model).

## Contributing

Pull requests and issues are welcome! If you see a novel technique or feature you think will make FLUX.1 inference better or faster, let us know and we'll do our best to integrate it.

## License

The code in this repository is licensed under the [Apache-2.0 License](LICENSE).

FLUX.1 kontext [dev] falls under the [`FLUX.1 [dev]` Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev/blob/main/LICENSE.md).