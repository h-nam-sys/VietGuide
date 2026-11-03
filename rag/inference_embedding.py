"""Serving adapter without LlamaIndex; uses the existing BGE inference functions."""
from .gpu_runtime import read_active, make_session, encode_embeddings


class LocalInferenceEmbedding:
    def __init__(self):
        from transformers import AutoTokenizer
        directory, metadata = read_active()
        self._metadata = metadata['embedding']
        self._tokenizer = AutoTokenizer.from_pretrained(directory/'embedding', local_files_only=True)
        self._session = make_session(directory/'embedding'/'model.onnx',
            device_id=metadata['device_id'], precision=metadata['precision'])

    def get_query_embedding(self, query):
        return encode_embeddings(self._session, self._tokenizer, [query], self._metadata, 'query')[0]

    def get_text_embedding_batch(self, texts):
        # Match the previous LlamaIndex adapter's embed_batch_size=1.
        return [encode_embeddings(self._session, self._tokenizer, [text],
                self._metadata, 'text')[0] for text in texts]
