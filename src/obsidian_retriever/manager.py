from __future__ import annotations

import os
import tempfile
import zipfile
from typing import Optional, List, Tuple, Iterable, Dict

from obsidian_parser.parse import load_obsidian_with_filters

from .vector_db.vector_store import VectorStore
from .graph_db.db import init_db
from .graph_db.repository import NoteRepository, ChunkRepository
from .schemas import NoteRecord, ChunkRecord, NoteLinkRecord, ChunkLinkRecord
from .utils.hash import text_hash


# TODO: Advanced graph algorithms for retriever(PageRank, Shortest path, etc.)
class KnowledgeBaseManager:
    def __init__(
        self,
        db_url: str,
        host: str = "localhost",
        port: int = 6333,
        prefer_grpc: bool = False,
        model_name: str = "openai/text-embedding-3-large",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        vault_id: Optional[str] = None,
    ) -> None:
        print(model_name)
        init_db(db_url)
        self.note_repository = NoteRepository()
        self.chunk_repository = ChunkRepository()
        self.vault_id = vault_id
        self.vector_store = VectorStore(
            host,
            port,
            prefer_grpc,
            model_name,
            api_key=api_key,
            base_url=base_url,
            vault_id=vault_id,
        )
        self.vector_store._ensure_collections()

    def _qualify_note_id(self, path: str) -> str:
        if not self.vault_id:
            return path
        return f"{self.vault_id}:{path}"

    def _build_note_dict(self, path: str, content: str, max_chunk_size: int) -> Dict:
        from obsidian_parser.parse import build_hierarchical_chunks

        name = os.path.basename(path)
        hierarchical_structure = build_hierarchical_chunks(content, max_chunk_size)
        return {
            "name": name,
            "path": path,
            "children": hierarchical_structure["children"],
            "chunks": hierarchical_structure["chunks"],
        }

    def _resolve_target_note_ids(self, target_name: str, path_to_note_id: Dict[str, str]) -> List[str]:
        resolved_targets: List[str] = []

        if target_name in path_to_note_id:
            resolved_targets.append(path_to_note_id[target_name])

        if "." not in target_name:
            alt = f"{target_name}.md"
            if alt in path_to_note_id:
                resolved_targets.append(path_to_note_id[alt])

        basename = os.path.basename(target_name)
        for path_key, nid in path_to_note_id.items():
            if os.path.basename(path_key) == basename and nid not in resolved_targets:
                resolved_targets.append(nid)

        return resolved_targets

    def upsert_note_from_content(self, path: str, content: str, max_chunk_size: int = 500) -> str:
        note_dict = self._build_note_dict(path, content, max_chunk_size)
        note_id = self._qualify_note_id(path)

        chunks_for_note, links_for_note, anchors_for_note = self._flatten_note_chunks(note_id, note_dict)

        existing_note = self.note_repository.get_by_id(note_id)
        if existing_note and existing_note.chunk_ids:
            self.vector_store.delete_chunks(existing_note.chunk_ids)

        note_record = NoteRecord(
            note_id=note_id,
            path=path,
            title=note_dict["name"],
            block_ids=None,
            chunk_ids=[c.chunk_id for c in chunks_for_note],
            links_to_notes=None,
        )
        self.add_note(note_record, chunks_for_note)

        all_notes = self.note_repository.list_all()
        path_to_note_id = {note.path: note.note_id for note in all_notes}
        path_to_note_id[path] = note_id

        anchor_to_chunk = {(note_id, title): chunk_id for title, chunk_id in anchors_for_note.items()}

        to_note_ids = set()
        chunk_links_map: Dict[str, List[str]] = {}

        for from_chunk_id, links in links_for_note.items():
            for link in links:
                if link.get("type") != "note":
                    continue
                target_name = link.get("link")
                if not target_name:
                    continue

                resolved_targets = self._resolve_target_note_ids(target_name, path_to_note_id)

                for to_note_id in resolved_targets:
                    to_note_ids.add(to_note_id)
                    anchor = link.get("anchor")
                    if anchor:
                        target_chunk_id = anchor_to_chunk.get((to_note_id, anchor))
                        if target_chunk_id:
                            chunk_links_map.setdefault(from_chunk_id, []).append(target_chunk_id)

        if to_note_ids:
            self.note_repository.set_links(note_id, list(to_note_ids), link_type="wiki")
        else:
            self.note_repository.set_links(note_id, [], link_type="wiki")

        for from_chunk_id, to_chunk_ids in chunk_links_map.items():
            self.chunk_repository.set_links(from_chunk_id, to_chunk_ids, link_type="reference")

        return note_id

    def delete_note_by_path(self, path: str) -> None:
        note_id = self._qualify_note_id(path)
        note = self.note_repository.get_by_id(note_id)
        if note and note.chunk_ids:
            self.vector_store.delete_chunks(note.chunk_ids)
        self.vector_store.delete_notes([note_id])
        self.note_repository.delete(note_id)

    def search_chunks(
        self,
        query: str,
        top_k: int = 10,
        note_ids: Optional[List[str]] = None,
        block_ids: Optional[List[str]] = None,
    ) -> List[Tuple[ChunkRecord, float]]:
        scored_points = self.vector_store.search_chunks(
            query=query, top_k=top_k, note_ids=note_ids, block_ids=block_ids
        )

        result = []
        for point in scored_points:
            chunk_id = point.payload.get("chunk_id") if point.payload else None
            if not chunk_id:
                continue
            chunk = self.chunk_repository.get_by_id(chunk_id)
            if not chunk:
                continue
            result.append((chunk, point.score))
        return result

    def search_notes(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[NoteRecord, float]]:
        scored_points = self.vector_store.search_notes(
            query=query, top_k=top_k
        )

        result = []
        for point in scored_points:
            note_id = point.payload.get("note_id") if point.payload else None
            if not note_id:
                continue
            note = self.note_repository.get_by_id(note_id)
            if not note:
                continue
            result.append((note, point.score))
        return result

    def get_note(self, note_id: str) -> Optional[NoteRecord]:
        note = self.note_repository.get_by_id(note_id)
        return note

    def get_chunk(self, chunk_id: str) -> Optional[ChunkRecord]:
        chunk = self.chunk_repository.get_by_id(chunk_id)
        return chunk

    def get_note_neighbors(self, note_id: str) -> Tuple[List[NoteLinkRecord], List[NoteLinkRecord]]:
        outgoing = self.note_repository.get_outgoing_links(note_id) or []
        incoming = self.note_repository.get_incoming_links(note_id) or []
        return outgoing, incoming

    def get_chunk_neighbors(self, chunk_id: str) -> Tuple[List[ChunkLinkRecord], List[ChunkLinkRecord]]:
        outgoing = self.chunk_repository.get_outgoing_links(chunk_id) or []
        incoming = self.chunk_repository.get_incoming_links(chunk_id) or []
        return outgoing, incoming

    def add_note(
        self,
        note: NoteRecord,
        chunks: Iterable[ChunkRecord],
    ) -> NoteRecord:
        chunks = list(chunks)

        chunks_content = "\n".join(chunk.text for chunk in chunks)
        content_hash = text_hash(chunks_content)

        note_db = self.note_repository.upsert(note, content_hash)

        stored_chunks = self.chunk_repository.replace_for_note(note_db.note_id, chunks)

        self.vector_store.upsert_chunks(stored_chunks)

        note_embedding = self.vector_store.aggregate_note_embedding(stored_chunks)
        self.vector_store.upsert_note(note_db, note_embedding)
        return note_db

    def set_note_links(
        self,
        from_note_id: str,
        to_note_ids: List[str],
        link_type: str = "wiki",
    ) -> None:
        self.note_repository.set_links(from_note_id, to_note_ids, link_type)

    def connect_notes(
        self,
        from_note_id: str,
        to_note_id: str,
        link_type: str = "wiki",
    ) -> None:
        self.note_repository.add_link(from_note_id, to_note_id, link_type)

    def set_chunk_links(
        self,
        from_chunk_id: str,
        to_chunk_ids: List[str],
        link_type: str = "reference",
    ) -> None:
        self.chunk_repository.set_links(from_chunk_id, to_chunk_ids, link_type)

    def connect_chunks(
        self,
        from_chunk_id: str,
        to_chunk_id: str,
        link_type: str = "reference",
    ) -> None:
        self.chunk_repository.add_link(from_chunk_id, to_chunk_id, link_type)

    # TODO: Note Tree hierarchal logic
    def init_vault_from_zip(
        self,
        zip_path: str,
        include_paths: List[str] = [],
        exclude_paths: List[str] = [],
        max_chunk_size: int = 500,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmpdir)

            notes_data = load_obsidian_with_filters(
                vault_path=tmpdir,
                include_paths=include_paths,
                exclude_paths=exclude_paths,
            )

            # путь файла (metadata["source"]) -> note_id (используем path)
            path_to_note_id = {}
            for note_dict in notes_data:
                path = note_dict["path"]
                note_id = self._qualify_note_id(path)  # соглашение: note_id == относительный путь файла
                path_to_note_id[path] = note_id

            # --- ПЕРВЫЙ ПРОХОД: чанки + якоря в памяти ---

            note_chunks = {}
            chunk_links_raw = {}
            anchor_to_chunk = {}
            chunk_to_note = {}

            for note_dict in notes_data:
                path = note_dict["path"]
                note_id = path_to_note_id[path]

                chunks_for_note, links_for_note, anchors_for_note = self._flatten_note_chunks(note_id, note_dict)

                note_chunks[note_id] = chunks_for_note

                for chunk_id, links in links_for_note.items():
                    chunk_links_raw[chunk_id] = links
                    chunk_to_note[chunk_id] = note_id

                for anchor_title, chunk_id in anchors_for_note.items():
                    anchor_to_chunk[(note_id, anchor_title)] = chunk_id

            # --- ВТОРОЙ ПРОХОД: пишем в граф + Qdrant ---

            for note_dict in notes_data:
                name = note_dict["name"]
                path = note_dict["path"]
                note_id = path_to_note_id[path]

                chunks_for_note = note_chunks[note_id]

                note_record = NoteRecord(
                    note_id=note_id,
                    path=path,
                    title=name,
                    block_ids=None,                      # блоков нет, плоская нота
                    chunk_ids=[c.chunk_id for c in chunks_for_note],
                    links_to_notes=None,
                )

                self.add_note(note_record, chunks_for_note)

            # --- ТРЕТИЙ ПРОХОД: LINKS_TO и REFERS_TO ---

            self._build_links_from_wikilinks(
                chunk_links_raw=chunk_links_raw,
                chunk_to_note=chunk_to_note,
                path_to_note_id=path_to_note_id,
                anchor_to_chunk=anchor_to_chunk,
            )

    # ============================
    # PRIVATE: плоское разворачивание дерева в чанки
    # ============================

    def _flatten_note_chunks(
        self,
        note_id: str,
        note_dict: dict,
    ) -> Tuple[List[ChunkRecord], Dict[str, List[dict]], Dict[str, str]]:
        """
        Обходит иерархию парсера и строит:
          - список всех ChunkRecord (root + под заголовками)
          - dict: chunk_id -> список сырых wiki-ссылок
          - dict: anchor_title -> chunk_id (первый чанк каждой секции)

        Блоки (BlockNode) НЕ создаём — всё плоско.
        """
        all_chunks = []
        chunk_links = {}
        anchors = {}

        chunk_counter = 0

        # root-чанки (текст вне заголовков)
        for ch in note_dict.get("chunks", []):
            chunk_id = f"{note_id}#chunk-{chunk_counter}"
            rec = ChunkRecord(
                chunk_id=chunk_id,
                note_id=note_id,
                block_id=None,
                index=chunk_counter,
                text=ch["data"],
                links_to_chunks=None,
            )
            all_chunks.append(rec)
            chunk_links[chunk_id] = ch.get("links", []) or []
            chunk_counter += 1

        # блоки (заголовки) и их поддеревья
        def process_block(node: dict) -> Optional[str]:
            """
            Возвращает chunk_id первого чанка этой секции (для anchor).
            """
            nonlocal chunk_counter

            title = node["title"]
            first_chunk_id = None

            # чанки прямо под этим заголовком
            for ch in node.get("chunks", []):
                chunk_id = f"{note_id}#chunk-{chunk_counter}"
                if first_chunk_id is None:
                    first_chunk_id = chunk_id

                rec = ChunkRecord(
                    chunk_id=chunk_id,
                    note_id=note_id,
                    block_id=None,      # плоская модель, блоков нет
                    index=chunk_counter,
                    text=ch["data"],
                    links_to_chunks=None,
                )
                all_chunks.append(rec)
                chunk_links[chunk_id] = ch.get("links", []) or []
                chunk_counter += 1

            # дочерние блоки
            for child in node.get("children", []):
                child_first = process_block(child)
                if first_chunk_id is None and child_first is not None:
                    first_chunk_id = child_first

            # если у секции вообще есть какой-то текст — регистрируем якорь
            if first_chunk_id is not None:
                anchors[title] = first_chunk_id

            return first_chunk_id

        for child_block in note_dict.get("children", []):
            process_block(child_block)

        return all_chunks, chunk_links, anchors

    # ============================
    # PRIVATE: восстановление ссылок
    # ============================

    def _build_links_from_wikilinks(
        self,
        chunk_links_raw: Dict[str, List[dict]],
        chunk_to_note: Dict[str, str],
        path_to_note_id: Dict[str, str],
        anchor_to_chunk: Dict[Tuple[str, str], str],
    ) -> None:
        """
        Строит:
          - Note -> Note LINKS_TO (doc-level)
          - Chunk -> Chunk REFERS_TO (по [[Note#Heading]])
        """
        note_links_pairs = set()
        chunk_links_pairs = set()

        for from_chunk_id, links in chunk_links_raw.items():
            from_note_id = chunk_to_note[from_chunk_id]

            for link in links:
                if link.get("type") != "note":
                    continue

                target_name = link.get("link")
                if not target_name:
                    continue

                resolved_targets = self._resolve_target_note_ids(target_name, path_to_note_id)

                # If no targets found - ignore (per new behaviour)
                if not resolved_targets:
                    continue

                # For each resolved target note, add a doc-level link
                for to_note_id in resolved_targets:
                    note_links_pairs.add((from_note_id, to_note_id))

                    # chunk-level: [[Note#Heading]]
                    anchor = link.get("anchor")
                    if anchor:
                        key = (to_note_id, anchor)
                        target_chunk_id = anchor_to_chunk.get(key)
                        if target_chunk_id:
                            chunk_links_pairs.add((from_chunk_id, target_chunk_id))

                # [[Note^block-ref]] пока игнорируем: мы не ведём mapping block_ref -> chunk

        # создаём Note -> Note
        for from_note_id, to_note_id in note_links_pairs:
            try:
                self.note_repository.add_link(from_note_id, to_note_id, link_type="wiki")
            except KeyError:
                # если какой-то note_id не найден — пропускаем
                continue

        # создаём Chunk -> Chunk
        for from_chunk_id, to_chunk_id in chunk_links_pairs:
            try:
                self.chunk_repository.add_link(from_chunk_id, to_chunk_id, link_type="reference")
            except KeyError:
                continue
