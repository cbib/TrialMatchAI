from trialmatchai.models.embedding.text_embedder import TextEmbedder, TextEmbedderConfig


class SecondLevelSentenceEmbedder(TextEmbedder):
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        use_gpu: bool = True,
        use_fp16: bool = False,
        max_length: int = 512,
        batch_size: int = 32,
        normalize: bool = True,
    ):
        super().__init__(
            TextEmbedderConfig(
                model_name=model_name,
                pooling="mean",
                max_length=max_length,
                batch_size=batch_size,
                use_gpu=use_gpu,
                use_fp16=use_fp16,
                normalize=normalize,
            )
        )
