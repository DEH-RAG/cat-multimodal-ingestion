<h1 align="center" id="title">Grinning Cat Multimodal Ingestion</h1>

<p id="description">This is a core plugin for the <a href="https://github.com/Grinning-Cat-RAG/grinning-cat-core">Grinning Cat</a> Project which owns the whole image lifecycle: it extracts the images produced by multimodal parsers during ingestion, embeds and stores them as image points, cascades their deletion with the source file, and attaches the recalled images to the LLM prompt on chat turns.</p>

<h2>🧐 What it does</h2>

Every image concern the core used to keep in `RabbitHole`, `StrayCat` and the file-manager route lives here, behind four hooks:

| Hook | Role |
| --- | --- |
| `rabbithole_collects_document_images` | Extracts the images from the parsed documents, **before** chunking |
| `rabbithole_stores_image_points` | Embeds the images, saves them as files and builds their memory points |
| `before_file_manager_file_delete` | Cascade-removes the extracted image files when a source file is deleted |
| `before_agentic_workflow` | Attaches the recalled images to the LLM prompt on a chat turn |

With the plugin disabled the core returns to upstream parity: no image collection, no image points, no image-file cascade, no image recall.

<h2>🖼️ Ingestion</h2>

Multimodal parsers (e.g. the PLUS `UnstructuredParser` with `extract_image_block_to_payload=True`) attach each extracted image to the parsed `Document` metadata as `image_base64` + `image_mime_type`. The plugin:

1. collects one entry per image **before chunking** — chunkers may drop that metadata — and strips the transient `image_base64` payload from the documents, so it never reaches the text chunk metadata nor the vector database;
2. embeds the images with the active embedder's `embed_images`, dispatched through the `run_in_ingestion_executor` hook (the dedicated lane of the *efficient ingestion* plugin, with a fallback to `asyncio.to_thread`);
3. saves each image as a file named deterministically after its source (`<source-stem>_img_<index>_<sha256[:8]><ext>`), so the association between an image and the document it came from is recoverable by name;
4. returns one `PointStruct` per image, carrying `image=True`, `image_file`, `source`, `when`, `hash` and, for chat-scoped ingestion, `chat_id`. The core appends them to the same collection as the text chunks.

An image whose embedding fails or is skipped (a `None` vector placeholder) is dropped entirely: neither saved as a file nor stored as a point. When the **source itself is an image**, the file is embedded as a single whole-image point and the parser's sub-crops are ignored.

Everything is gated on the active embedder being multimodal: the check resolves the embedder to its settings class through the lizard's `ServiceFactory` and asks `is_multimodal()`, falling back to an instance-level check for custom embedders not registered in the factory.

<h2>🗑️ Deletion cascade</h2>

When a source file is deleted, the hook fires after the file is removed and **before** its memory points are — the point metadata still holds the image file names. The image points of that source are paged through and every `image_file` they reference is removed from the agent storage, so no orphan image lingers on disk.

<h2>👁️ Recall</h2>

On a chat turn, when the active embedder is multimodal and the LLM is vision-capable (`llm_vision_capable` hook), the image points recalled into the working memory become full LangChain content parts — `{"type": "image_url", "image_url": {"url": "<data-uri>"}}` — attached to the agentic workflow task, so the model actually sees them.

Two caps bound one turn: at most **5 images** (`MAX_IMAGES_PER_TURN`) and **5 MiB** in total (`MAX_IMAGE_TOTAL_BYTES`). The mime type comes from the point metadata, sniffed with Pillow only when it is missing. The whole path is best-effort: on any failure it returns no images and the `[Image]` text placeholder already in the context keeps the turn working text-only.

<h2>🤝 Integration with the efficient ingestion engine (EffING)</h2>

The image lifecycle is usable identically by both ingestion engines — the core CAT engine (RabbitHole, "old embedding") and the efficient-ingestion engine (EffING) — behind the same hook-driven contract. The pure helpers in `cat.plugins.cat_multimodal_ingestion.ingestion` are the stable import surface:

| Symbol | Role |
| --- | --- |
| `collect_document_images` / `strip_image_payload` | Extraction of the parser images before chunking |
| `image_file_name` | Deterministic image file naming (`<source-stem>_img_<index>_<sha256[:8]><ext>`) |
| `delete_source_image_files` | Deletion cascade for a source's image files |
| `is_multimodal_embedder_active` | Multimodal-embedder detection |
| `build_image_points` | Image-point building (bivalent, see below) |

`build_image_points` is bivalent: `vectors="embed"|"empty"` embeds the images with the active embedder or builds payload-only points (`vector={}`) for a later `embedding` phase; `on_embed_failure="drop"|"keep_empty"` drops a failed embed entirely or keeps it payload-only for a retry. The defaults (`vectors="embed"`, `on_embed_failure="drop"`) are the core behavior.

**Gate-in-callers rule:** the pure functions never gate. The core hook (`rabbithole_collects_document_images`) gates extraction on `is_multimodal_embedder_active`; the EffING engine extracts always and defers the decision to its `embedding` phase.

**EffING-side dedup contract:** the EffING `reembed.py` image parts are expected to call the MmodING helpers instead of re-implementing them locally — `_store_empty_vectors` (reembed.py:200) image part → `build_image_points(vectors="empty")`; `_embed_phase` (reembed.py:258) image part → `build_image_points(vectors="embed", on_embed_failure="keep_empty")`; `_cleanup_orphan_images` (reembed.py:102) → `delete_source_image_files`.

EffING's `ingestion_phase_specs` phase-registration API (live 2026-09-19) is deliberately NOT used by this plugin: the image lifecycle spans the two built-in phases (`parsing_chunking` + `embedding`), and registering a phase would require persisting transient parser images between phases (anti-pattern). The provider pattern (spec + marker + run) documented in EffING's README is the reference if a multimodal-settings marker is ever desired.

<h2>🛠️ Installation:</h2>

<p>1. Clone this repo and copy it on cat plugins folder</p>
<p>2. Install from admin panel on the [Grinning Cat Web Admin](https://github.com/matteocacciola/grinning-cat-admin)</p>

<h2>🛡️ License:</h2>
This project is licensed under the GNU GENERAL PUBLIC LICENSE
