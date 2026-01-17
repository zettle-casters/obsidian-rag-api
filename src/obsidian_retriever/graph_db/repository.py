from __future__ import annotations

from typing import List, Iterable, Optional

from neomodel import db

from .models import NoteNode, BlockNode, ChunkNode
from .mapper import (
    note_node_to_record, block_node_to_record, chunk_node_to_record,
    note_link_rel_to_record, chunk_link_rel_to_record
)
from ..schemas import NoteRecord, BlockRecord, ChunkRecord, NoteLinkRecord, ChunkLinkRecord


class NoteRepository:
    def get_by_id(self, note_id : str) -> Optional[NoteRecord]:
        node = NoteNode.nodes.get_or_none(note_id=note_id)
        if not node:
            return None
        return note_node_to_record(node)

    def get_by_path(self, path : str) -> Optional[NoteRecord]:
        node = NoteNode.nodes.get_or_none(path=path)
        if not node:
            return None
        return note_node_to_record(node)

    def list_all(self) -> List[NoteRecord]:
        return [note_node_to_record(node) for node in NoteNode.nodes.all()]

    def upsert(self, note : NoteRecord, content_hash : str) -> NoteRecord:
        node = NoteNode.nodes.get_or_none(note_id=note.note_id)
        if node is None:
            node = NoteNode(
                note_id=note.note_id,
                path=note.path,
                title=note.title,
                content_hash=content_hash
            ).save()
        else:
            node.path = note.path
            node.title = note.title
            node.content_hash = content_hash
            node.save()
        return note_node_to_record(node)

    def add_link(self, from_note_id : str, to_note_id : str, link_type : str = "wiki") -> None:
        from_node = NoteNode.nodes.get_or_none(note_id=from_note_id)
        if not from_node:
            raise KeyError(f"Note with id {from_note_id} not found")

        to_node = NoteNode.nodes.get_or_none(note_id=to_note_id)
        if not to_node:
            raise KeyError(f"Note with id {to_note_id} not found")

        from_node.outgoing_links.connect(to_node, {"link_type" : link_type})

    @db.transaction
    def set_links(self, from_note_id : str, to_note_ids : List[str], link_type : str = "wiki") -> None:
        from_node = NoteNode.nodes.get_or_none(note_id=from_note_id)
        if not from_node:
            raise KeyError(f"Note with id {from_note_id} not found")

        for target in from_node.outgoing_links.all():
            from_node.outgoing_links.disconnect(target)

        for to_id in to_note_ids:
            to_node = NoteNode.nodes.get_or_none(note_id=to_id)
            if not to_node:
                raise KeyError(f"Note with id {to_id} not found")
            from_node.outgoing_links.connect(to_node, {'link_type': link_type})

    def get_outgoing_links(self, note_id : str) -> Optional[List[NoteLinkRecord]]:
        node = NoteNode.nodes.get_or_none(note_id=note_id)
        if not node:
            return None

        result = []
        for target in node.outgoing_links.all():
            rel = node.outgoing_links.relationship(target)
            result.append(note_link_rel_to_record(node, target, rel))
        return result

    def get_incoming_links(self, note_id : str) -> Optional[List[NoteLinkRecord]]:
        node = NoteNode.nodes.get_or_none(note_id=note_id)
        if not node:
            return None

        result = []
        for source in node.incoming_links.all():
            rel = source.outgoing_links.relationship(node)
            result.append(note_link_rel_to_record(source, node, rel))
        return result

    @db.transaction
    def delete(self, note_id : str) -> None:
        note = NoteNode.nodes.get_or_none(note_id=note_id)
        if not note:
            return

        for chunk in note.chunks.all():
            chunk.delete()

        for root_block in note.blocks.all():
            self._delete_block_subtree(root_block)

        note.delete()

    def _delete_block_subtree(self, block: BlockNode) -> None:
        for chunk in block.chunks.all():
            chunk.delete()

        for child in block.children.all():
            self._delete_block_subtree(child)

        block.delete()

class BlockRepository:
    def get_by_id(self, block_id : str) -> Optional[BlockRecord]:
        block = BlockNode.nodes.get_or_none(block_id=block_id)
        if not block:
            return None
        return block_node_to_record(block)

    def get_root_blocks(self, note_id : str) -> List[BlockRecord]:
        note = NoteNode.nodes.get_or_none(note_id=note_id)
        if not note:
            raise KeyError(f"Note with id {note_id} not found")

        blocks = note.blocks.all()
        blocks_sorted = sorted(blocks, key=lambda block : block.index)
        return [block_node_to_record(block) for block in blocks_sorted]

    def get_children(self, block_id : str) -> List[BlockRecord]:
        block = BlockNode.nodes.get_or_none(block_id=block_id)
        if not block:
            raise KeyError(f"Block with id {block_id} not found")

        children = block.children.all()
        children_sorted = sorted(children, key=lambda child : child.index)
        return [block_node_to_record(child) for child in children_sorted]

    @db.transaction
    def create_block(
        self,
        note_id : str,
        title : str,
        parent_block_id : Optional[str] = None
    ) -> BlockRecord:
        note = NoteNode.nodes.get_or_none(note_id=note_id)
        if not note:
            raise KeyError(f"Note with id {note_id} not found")

        parent_block = None
        if parent_block_id is not None:
            parent_block = BlockNode.nodes.get_or_none(block_id=parent_block_id)
            if not parent_block:
                raise KeyError(f"Block with id {parent_block_id} not found")
            parent_children_count = len(parent_block.children.all())
            index = parent_block.index + parent_children_count + 1
        else:
            index = len(note.blocks.all()) + 1

        block_id = f"{note_id}#block-{index}"

        block_node = BlockNode(
            block_id=block_id,
            note_id=note_id,
            index=index,
            title=title,
        ).save()

        if parent_block is None:
            note.blocks.connect(block_node)
        else:
            parent_block.children.connect(block_node)
        return block_node_to_record(block_node)


class ChunkRepository:
    def get_by_id(self, chunk_id : str) -> Optional[ChunkRecord]:
        chunk = ChunkNode.nodes.get_or_none(chunk_id=chunk_id)
        if not chunk:
            return None
        return chunk_node_to_record(chunk)

    def get_by_note(self, note_id : str) -> List[ChunkRecord]:
        note = NoteNode.nodes.get_or_none(note_id=note_id)
        if not note:
            raise KeyError(f"Note with id {note_id} not found")

        chunks = note.chunks.all()
        chunks_sorted = sorted(chunks, key=lambda chunk : chunk.index)
        return [chunk_node_to_record(chunk) for chunk in chunks_sorted]

    def get_by_block(self, block_id : str) -> List[ChunkRecord]:
        block = BlockNode.nodes.get_or_none(block_id=block_id)
        if not block:
            raise KeyError(f"Block with id {block_id} not found")

        chunks = block.chunks.all()
        chunks_sorted = sorted(chunks, key=lambda chunk : chunk.index)
        return [chunk_node_to_record(chunk) for chunk in chunks_sorted]

    @db.transaction
    def add_to_block(self, block_id : str, chunks : Iterable[ChunkRecord]) -> List[ChunkRecord]:
        block = BlockNode.nodes.get_or_none(block_id=block_id)
        if not block:
            raise KeyError(f"Block with id {block_id} not found")

        chunks = list(chunks)
        if not chunks:
            return []

        results = []
        for chunk in chunks:
            if chunk.note_id != block.note_id:
                raise ValueError(f"Chunk note_id={chunk.note_id} does not match block.note_id={block.note_id}")

            chunk_node = ChunkNode(
                chunk_id=chunk.chunk_id,
                note_id=chunk.note_id,
                index=chunk.index,
                text=chunk.text
            ).save()
            block.chunks.connect(chunk_node)
            results.append(chunk_node_to_record(chunk_node))

        return results


    def add_link(self, from_chunk_id : str, to_chunk_id : str, link_type : str = "reference") -> None:
        from_chunk = ChunkNode.nodes.get_or_none(chunk_id=from_chunk_id)
        if not from_chunk:
            raise KeyError(f"Chunk with id {from_chunk_id} not found")

        to_chunk = ChunkNode.nodes.get_or_none(chunk_id=to_chunk_id)
        if not to_chunk:
            raise KeyError(f"Chunk with id {to_chunk_id} not found")

        from_chunk.outgoing_links.connect(to_chunk, {"link_type" : link_type})

    @db.transaction
    def set_links(self, from_chunk_id : str, to_chunk_ids : List[str], link_type : str = "reference") -> None:
        from_chunk = ChunkNode.nodes.get_or_none(chunk_id=from_chunk_id)
        if not from_chunk:
            raise KeyError(f"Chunk with id {from_chunk_id} not found")

        for target in from_chunk.outgoing_links.all():
            from_chunk.outgoing_links.disconnect(target)

        for to_id in to_chunk_ids:
            to_chunk = ChunkNode.nodes.get_or_none(chunk_id=to_id)
            if not to_chunk:
                raise KeyError(f"Chunk with id {to_id} not found")
            from_chunk.outgoing_links.connect(to_chunk, {'link_type': link_type})

    @db.transaction
    def replace_for_note(self, note_id : str, chunks : Iterable[ChunkRecord]) -> List[ChunkRecord]:
        note = NoteNode.nodes.get_or_none(note_id=note_id)
        if not note:
            raise KeyError(f"Note with id {note_id} not found")

        for old_chunk in note.chunks.all():
            old_chunk.delete()

        for chunk in chunks:
            chunk_node = ChunkNode(
                chunk_id=chunk.chunk_id,
                note_id=chunk.note_id,
                index=chunk.index,
                text=chunk.text,
            ).save()
            note.chunks.connect(chunk_node)

        new_chunks = note.chunks.all()
        new_chunks_sorted = sorted(new_chunks, key=lambda c: c.index)
        return [chunk_node_to_record(c) for c in new_chunks_sorted]

    def get_outgoing_links(self, chunk_id : str) -> Optional[List[ChunkLinkRecord]]:
        chunk = ChunkNode.nodes.get_or_none(chunk_id=chunk_id)
        if not chunk:
            return None

        result = []
        for target in chunk.outgoing_links.all():
            rel = chunk.outgoing_links.relationship(target)
            result.append(chunk_link_rel_to_record(chunk, target, rel))
        return result

    def get_incoming_links(self, chunk_id : str) -> Optional[List[ChunkLinkRecord]]:
        chunk = ChunkNode.nodes.get_or_none(chunk_id=chunk_id)
        if not chunk:
            return None

        result = []
        for source in chunk.incoming_links.all():
            rel = source.outgoing_links.relationship(chunk)
            result.append(chunk_link_rel_to_record(source, chunk, rel))
        return result
