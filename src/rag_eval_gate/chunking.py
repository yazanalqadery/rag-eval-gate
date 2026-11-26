"""Split Markdown documents into chunks, one per '## ' section."""

def chunk_by_section(text: str) -> list[str]:
    """Return one chunk per '## ' heading. The heading stays as the chunk's first line.
    '### ' subheadings stay inside their parent section."""
    chunks = []
    current = []  # lines of the chunk we're building
    for line in text.splitlines():
        if line.startswith("## "):
            if current:  # finish the previous chunk first
                chunks.append("\n".join(current).strip())
            current = [line[3:]]  # start a new chunk with the heading, without "## "
        else:
            current.append(line)
    if current:  # don't forget the last chunk
        chunks.append("\n".join(current).strip())
    return [chunk for chunk in chunks if chunk]

def chunk_title(chunk: str) -> str:
    return chunk.splitlines()[0]