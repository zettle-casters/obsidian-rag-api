from .models import NoteNode, BlockNode, ChunkNode, NoteLinkRel, ChunkLinkRel
from ..schemas import NoteRecord, BlockRecord, ChunkRecord, NoteLinkRecord, ChunkLinkRecord

def note_node_to_record(note: NoteNode) -> NoteRecord:
    block_ids = [block.block_id for block in note.blocks.all()]
    chunk_ids = [chunk.chunk_id for chunk in note.chunks.all()]
    links_to_notes = [ref_note.note_id for ref_note in note.outgoing_links.all()]

    return NoteRecord(
        note_id=note.note_id,
        path=note.path,
        title=note.title,
        block_ids=block_ids or None,
        chunk_ids=chunk_ids or None,
        links_to_notes=links_to_notes or None,
    )

def block_node_to_record(block: BlockNode) -> BlockRecord:
    parents = block.parent.all()
    parent_block_id = parents[0].block_id if parents else None
    child_block_ids = [child.block_id for child in block.children.all()]
    chunk_ids = [chunk.chunk_id for chunk in block.chunks.all()]

    return BlockRecord(
        block_id=block.block_id,
        note_id=block.note_id,
        index=block.index,
        parent_block_id=parent_block_id,
        child_block_ids=child_block_ids or None,
        chunk_ids=chunk_ids or None,
    )

def chunk_node_to_record(chunk: ChunkNode) -> ChunkRecord:
    links_to_chunks = [ref_chunk.chunk_id for ref_chunk in chunk.outgoing_links.all()]
    block_id = chunk.block.all()[0].block_id if chunk.block else None

    return ChunkRecord(
        chunk_id=chunk.chunk_id,
        note_id=chunk.note_id,
        block_id=block_id,
        index=chunk.index,
        text=chunk.text,
        links_to_chunks=links_to_chunks or None,
    )

def note_link_rel_to_record(from_note: NoteNode, to_note: NoteNode, rel: NoteLinkRel) -> NoteLinkRecord:
    return NoteLinkRecord(
        from_note_id=from_note.note_id,
        to_note_id=to_note.note_id,
        link_type=rel.link_type or "wiki",
    )

def chunk_link_rel_to_record(from_chunk: ChunkNode, to_chunk: ChunkNode, rel: ChunkLinkRel) -> ChunkLinkRecord:
    return ChunkLinkRecord(
        from_chunk_id=from_chunk.chunk_id,
        to_chunk_id=to_chunk.chunk_id,
        link_type=rel.link_type or "reference",
    )
