import re
import json
import os
import argparse
from typing import Dict, List
from urllib.parse import unquote
from langchain_community.document_loaders import ObsidianLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_wiki_links(text: str) -> List[Dict]:
    """
    Извлекает все вики-ссылки Obsidian из текста с поддержкой:
    - [[File]]
    - [[File|Alias]]
    - [[File#Section]]
    - [[File#Section|Alias]]
    - [[File^block-ref]]
    - ![[Embedded File]]
    """

    # Основной паттерн для всех типов вики-ссылок
    pattern = r"(!?)\[\[([^\]]+?)(?:\|([^\]]+?))?\]\]"

    links = []

    for match in re.finditer(pattern, text):
        is_embedded = match.group(1) == "!"  # ![[ - embedded файл
        link_content = match.group(2).strip()  # Основное содержимое ссылки
        alias = match.group(3) if match.group(3) else None

        # Разбираем link_content на компоненты
        base_link = link_content
        anchor = None
        block_ref = None

        # Проверяем на наличие якоря # или блочной ссылки ^
        if "#" in link_content:
            base_link, anchor = link_content.split("#", 1)
        elif "^" in link_content:
            base_link, block_ref = link_content.split("^", 1)

        base_link = unquote(base_link)

        # Определяем тип ссылки
        if anchor and "|" in anchor:
            # Случай [[File#Section|Alias]]
            anchor_parts = anchor.split("|", 1)
            anchor = anchor_parts[0]
            if not alias:
                alias = anchor_parts[1]

        link_info = {
            "raw": match.group(0),  # Полная исходная строка
            "is_embedded": is_embedded,
            "link": base_link,  # Основная часть ссылки (без якоря/блока)
            "anchor": anchor,  # #section часть или None
            "block_ref": block_ref,  # ^block-id часть или None
            "alias": alias if alias else base_link,  # Отображаемый текст
        }

        # Определяем тип контента по расширению
        if "." in base_link:
            ext = base_link.split(".")[-1].lower()
            link_info["type"] = (
                "image"
                if ext in ["png", "jpg", "jpeg", "gif", "bmp", "svg"]
                else "file"
            )
            link_info["extension"] = ext
        else:
            link_info["type"] = "note"
            link_info["extension"] = "md"

        links.append(link_info)

    return links


def create_chunks_for_text(text: str, max_chunk_size: int = 500) -> List[Dict]:
    """Функция для разбиения текста на чанки с заданным размером """
    if not text.strip():
        return []

    # Создаем сплиттер с нужными параметрами
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size, chunk_overlap=0
    )

    # Разбиваем текст
    text_chunks = splitter.split_text(text.strip())

    # Преобразуем в нужный формат
    chunks = []
    for chunk_text in text_chunks:
        if chunk_text.strip():  # Проверяем, что чанк не пустой
            chunks.append({
                "data": chunk_text,
                "links": extract_wiki_links(chunk_text)})

    return chunks


# Функция для построения иерархического дерева с чанками в листьях
def build_hierarchical_chunks(text: str, max_chunk_size: int = 500) -> Dict:
    """
    Строит дерево где:
    - корень: документ
    - узлы: блоки с заголовками
    - листья: чанки с контентом
    """
    lines = text.split("\n")

    # Корневой узел - вся заметка
    root = {
        "level": 0,
        "chunks": [],  # чанки для текста без заголовков
        "children": [],  # дочерние блоки с заголовками
    }

    stack = [root]  # стек для отслеживания текущего уровня

    current_content_lines = []

    for line in lines:
        # Проверяем заголовок
        header_match = re.match(r"^(#{1,6})\s+(.+)$", line)

        if header_match:
            # Сохраняем накопленный контент в текущий узел
            if current_content_lines:
                content_text = "\n".join(current_content_lines).strip()
                if content_text:
                    node_chunks = create_chunks_for_text(content_text, max_chunk_size)
                    if node_chunks:
                        stack[-1]["chunks"].extend(node_chunks)
                current_content_lines = []

            level = len(header_match.group(1))
            title = header_match.group(2).strip()

            # Создаем новый узел (блок с заголовком)
            new_node = {
                "title": title,
                "level": level,
                "chunks": [],  # чанки для контента этого блока
                "children": [],  # вложенные блоки
            }

            # Находим правильного родителя
            while len(stack) > 1 and stack[-1]["level"] >= level:
                stack.pop()

            # Добавляем к родителю
            stack[-1]["children"].append(new_node)
            stack.append(new_node)

        else:
            # Накопливаем контент
            current_content_lines.append(line)

    # Обрабатываем оставшийся контент
    if current_content_lines:
        content_text = "\n".join(current_content_lines).strip()
        if content_text:
            node_chunks = create_chunks_for_text(content_text, max_chunk_size)
            if node_chunks:
                stack[-1]["chunks"].extend(node_chunks)

    return root


def load_obsidian_with_filters(
        vault_path: str,
        include_paths: List[str] = None,
        exclude_paths: List[str] = None) -> List[Dict]:
    """
    Загружает документы Obsidian с фильтрацией по путям
    """
    # Custom implementation to avoid ObsidianLoader's directory issue
    # Collect all .md files manually
    from pathlib import Path

    vault_path_obj = Path(vault_path)
    md_files = []

    for md_file in vault_path_obj.rglob("*.md"):
        # Skip if it's a directory (edge case where directory has .md extension)
        if md_file.is_dir():
            continue
        md_files.append(md_file)

    include_paths = list(map(
        lambda x: os.path.normpath(vault_path + x),
        include_paths)) if include_paths else []
    exclude_paths = list(map(
        lambda x: os.path.normpath(vault_path + x),
        exclude_paths)) if exclude_paths else []
    result = []

    for md_file in md_files:
        file_path = os.path.normpath(str(md_file))

        # Исключаем документы из exclude_paths
        if exclude_paths and any(file_path.startswith(path) for path in exclude_paths):
            continue

        # Включаем документы из include_paths (если заданы)
        if include_paths:
            if not any(file_path.startswith(path) for path in include_paths):
                continue

        # Read file content
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Warning: Failed to read {md_file}: {e}")
            continue

        hierarchical_structure = build_hierarchical_chunks(
            content, max_chunk_size=500
        )

        # Extract relative path from vault_path
        relative_path = str(md_file.relative_to(vault_path_obj))

        # Extract tags from frontmatter (basic implementation)
        tags = []
        if content.startswith('---'):
            try:
                end_idx = content.index('---', 3)
                frontmatter = content[3:end_idx]
                for line in frontmatter.split('\n'):
                    if line.strip().startswith('tags:'):
                        tag_str = line.split(':', 1)[1].strip()
                        # Handle both list format and comma-separated
                        if tag_str.startswith('['):
                            tags = [t.strip(' "\'[]') for t in tag_str.strip('[]').split(',')]
                        else:
                            tags = [t.strip() for t in tag_str.split(',')]
            except (ValueError, IndexError):
                pass

        result.append(
            {
                "name": md_file.name,
                "path": relative_path,
                "tags": tags,
                "children": hierarchical_structure["children"],
                "chunks": hierarchical_structure["chunks"],
            }
        )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Загрузка Obsidian vault с фильтрацией"
    )
    parser.add_argument(
        "--path",
        help="Путь к vault Obsidian",
        default="D:/Apps/Obsidian/vaults/obsidian_tech_wiki-main",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="result.json",
        help="Имя выходного JSON файла (по умолчанию: result3.json)",
    )
    args = parser.parse_args()

    result = load_obsidian_with_filters(
        vault_path=args.path,
        include_paths=[],
        exclude_paths=["/Attachments", "/Authors", "/Drawings", "/Templates"],
        # Пути указываются относительно корневой директории
    )

    with open(args.output, "w", encoding="UTF-8") as file:
        json.dump(result, file, indent=2)

    print(f"Результат сохранен в {args.output}")


if __name__ == "__main__":
    main()
