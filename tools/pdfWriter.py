import markdown
from weasyprint import HTML
import uuid

def write_pdf(markdown_text: str) -> str:
    """
    Use this function to create pdf file from markdown content.

    Args:
        markdown_text (str): The markdown content to convert to pdf.
    
    Returns:
        str: The path to the pdf file.
    """

    html_output = markdown.markdown(markdown_text)

    file_name = f"{uuid.uuid4()}.pdf"
    HTML(string=html_output).write_pdf(file_name)

    print(f"PDF file created: {file_name}")
    return f"/?param={file_name}"

