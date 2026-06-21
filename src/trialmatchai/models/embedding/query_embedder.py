from trialmatchai.models.embedding.text_embedder import TextEmbedder, TextEmbedderConfig


class QueryEmbedder(TextEmbedder):
    def __init__(
        self,
        model_name: str = "ncbi/MedCPT-Query-Encoder",
        max_length: int = 512,
        use_gpu: bool = True,
        use_fp16: bool = False,
        batch_size: int = 32,
        normalize: bool = True,
    ):
        super().__init__(
            TextEmbedderConfig(
                model_name=model_name,
                pooling="cls",
                max_length=max_length,
                batch_size=batch_size,
                use_gpu=use_gpu,
                use_fp16=use_fp16,
                normalize=normalize,
            )
        )
