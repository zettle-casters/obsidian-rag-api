import hashlib

def text_hash(text : str) -> str:
    hash = hashlib.new('sha256')
    hash.update(text.encode())
    return hash.hexdigest()

def id_from_text(text: str) -> int:
    hashed_text = text_hash(text)
    return int(hashed_text[:16], 16)
