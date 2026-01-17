from __future__ import annotations

from neomodel import StructuredNode, StructuredRel, RelationshipTo, RelationshipFrom
from neomodel.properties import StringProperty, IntegerProperty, DateTimeProperty, UniqueIdProperty


class NoteLinkRel(StructuredRel):
    link_type = StringProperty(default="wiki")

class TreeLinkRel(StructuredRel):
    link_type = StringProperty(default="note_tree")

class ChunkLinkRel(StructuredRel):
    link_type = StringProperty(default="reference")


class NoteNode(StructuredNode):
    uuid = UniqueIdProperty()

    note_id = StringProperty(unique_index=True, required=True)
    path = StringProperty(unique_index=True, required=True)
    title = StringProperty(required=True)
    content_hash = StringProperty(required=True)

    created_at = DateTimeProperty(default_now=True)
    updated_at = DateTimeProperty(default_now=True, update_now=True)

    chunks = RelationshipTo("ChunkNode", "HAS_CHUNK", model=TreeLinkRel)
    blocks = RelationshipTo("BlockNode", "HAS_ROOT_BLOCK", model=TreeLinkRel)

    outgoing_links = RelationshipTo("NoteNode", "LINKS_TO", model=NoteLinkRel)
    incoming_links = RelationshipFrom("NoteNode", "LINKS_TO", model=NoteLinkRel)

class BlockNode(StructuredNode):
    uuid = UniqueIdProperty()

    block_id = StringProperty(unique_index=True, required=True)
    note_id = StringProperty(required=True)
    index = IntegerProperty(required=True)
    title = StringProperty(required=True)

    note = RelationshipFrom("NoteNode", "HAS_ROOT_BLOCK", model=TreeLinkRel)

    parent = RelationshipFrom("BlockNode", "HAS_CHILD_BLOCK", model=TreeLinkRel)
    children = RelationshipTo("BlockNode", "HAS_CHILD_BLOCK", model=TreeLinkRel)

    chunks = RelationshipTo("ChunkNode", "HAS_CHUNK")

class ChunkNode(StructuredNode):
    uuid = UniqueIdProperty()

    chunk_id = StringProperty(unique_index=True, required=True)
    note_id = StringProperty(required=True)
    index = IntegerProperty(required=True)
    text = StringProperty(required=True)

    note = RelationshipFrom("NoteNode", "HAS_CHUNK", model=TreeLinkRel)
    block = RelationshipFrom("BlockNode", "HAS_CHUNK", model=TreeLinkRel)

    outgoing_links = RelationshipTo("ChunkNode", "REFERS_TO", model=ChunkLinkRel)
    incoming_links = RelationshipFrom("ChunkNode", "REFERS_TO", model=ChunkLinkRel)
