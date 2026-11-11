"""The parser ``image_base64`` payload is stripped whatever the embedder.

Regression for the vector-DB bloat: with a text embedder the
``rabbithole_collects_document_images`` hook used to return before
``strip_image_payload`` ran, so the payload reached the chunk metadata and the
stored text points. The hook must collect images only for a multimodal
embedder, but strip the docs in every case.

Standalone script (plugin rules): the host imports every ``.py`` under the
plugin folder, so the ENTIRE body lives inside ``if __name__ == "__main__":``
and importing this file executes zero code. Run it from the host checkout:

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-multimodal-ingestion/tests/test_strip_image_payload_unconditional.py
"""

if __name__ == "__main__":
    import asyncio
    import base64
    import os
    import sys

    os.environ.setdefault("CAT_JWT_SECRET", "test_only_jwt_secret_0123456789_abcdefghij")
    os.environ.setdefault("CAT_CRYPTO_KEY", "grinning_cat_key")
    os.environ.setdefault("CAT_CRYPTO_SALT", "grinning_cat_salt")

    from langchain_core.documents import Document

    from cat.plugins.cat_multimodal_ingestion import plugin as multimodal_plugin

    def _doc() -> Document:
        return Document(
            page_content="image element",
            metadata={
                "image_base64": base64.b64encode(b"hello").decode(),
                "image_mime_type": "image/png",
                "keep": "me",
            },
        )

    def _find_keys(value, key):
        found = []
        if isinstance(value, dict):
            for k, v in value.items():
                if k == key:
                    found.append(v)
                found.extend(_find_keys(v, key))
        elif isinstance(value, list):
            for v in value:
                found.extend(_find_keys(v, key))
        return found

    async def main() -> int:
        checks = 0
        original = multimodal_plugin.is_multimodal_embedder_active

        async def active(_cat) -> bool:
            return True

        async def inactive(_cat) -> bool:
            return False

        try:
            # Multimodal ON: the image is collected AND the payload is stripped.
            doc = _doc()
            multimodal_plugin.is_multimodal_embedder_active = active
            images = await multimodal_plugin.rabbithole_collects_document_images.function(
                [], [doc], object()
            )
            assert len(images) == 1, f"expected 1 collected image, got {images!r}"
            assert images[0]["image_bytes"] == b"hello", images[0]
            assert images[0]["image_mime_type"] == "image/png", images[0]
            assert "image_base64" not in doc.metadata, doc.metadata
            assert doc.metadata["keep"] == "me", doc.metadata
            checks += 4

            # Multimodal OFF: no image collected, incoming list unchanged, payload stripped.
            doc = _doc()
            incoming = [{"sentinel": True}]
            multimodal_plugin.is_multimodal_embedder_active = inactive
            images = await multimodal_plugin.rabbithole_collects_document_images.function(
                incoming, [doc], object()
            )
            assert images is incoming, f"incoming images must be returned unchanged: {images!r}"
            assert "image_base64" not in doc.metadata, doc.metadata
            assert doc.metadata["keep"] == "me", doc.metadata
            checks += 3

            # Real parser/chunker shape: the payload is nested in
            # source_chunks[*].metadata. The recursive strip removes every
            # image_base64 while preserving text, source_chunks and coordinates.
            nested = Document(
                page_content="text",
                metadata={
                    "source": "ud4.pdf",
                    "source_chunks": [
                        {
                            "text": "a",
                            "metadata": {
                                "image_base64": base64.b64encode(b"one").decode(),
                                "coordinates": {"x": 1},
                            },
                        },
                        {
                            "text": "b",
                            "metadata": {
                                "image_base64": base64.b64encode(b"two").decode(),
                                "coordinates": {"x": 2},
                            },
                        },
                    ],
                },
            )
            multimodal_plugin.strip_image_payload([nested])
            assert _find_keys(nested.metadata, "image_base64") == [], nested.metadata
            assert nested.metadata["source"] == "ud4.pdf", nested.metadata
            assert [c["text"] for c in nested.metadata["source_chunks"]] == ["a", "b"]
            assert nested.metadata["source_chunks"][0]["metadata"]["coordinates"] == {"x": 1}
            assert nested.metadata["source_chunks"][1]["metadata"]["coordinates"] == {"x": 2}
            checks += 5

            # No docs: the incoming list is returned untouched.
            assert (
                await multimodal_plugin.rabbithole_collects_document_images.function(
                    incoming, [], object()
                )
                is incoming
            )
            checks += 1
        finally:
            multimodal_plugin.is_multimodal_embedder_active = original

        print(f"test_strip_image_payload_unconditional: {checks} checks passed")
        return 0

    sys.exit(asyncio.run(main()))
