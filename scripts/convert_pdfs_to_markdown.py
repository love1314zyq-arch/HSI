import re
import subprocess
from pathlib import Path


ROOT = Path("/home/zyq/HSI/文献")
OUT_DIR = ROOT / "markdown"


def normalize_text(text: str) -> str:
    text = text.replace("\f", "\n\n---\n\n")
    lines = [line.rstrip() for line in text.splitlines()]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip() + "\n"


def convert_pdf(pdf_path: Path) -> Path:
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    content = normalize_text(result.stdout)
    md_name = pdf_path.with_suffix(".md").name
    md_path = OUT_DIR / md_name
    title = pdf_path.stem
    body = f"# {title}\n\nSource: `{pdf_path.name}`\n\n{content}"
    md_path.write_text(body, encoding="utf-8")
    return md_path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(ROOT.glob("*.pdf"))
    for pdf in pdfs:
        md_path = convert_pdf(pdf)
        print(md_path)


if __name__ == "__main__":
    main()
