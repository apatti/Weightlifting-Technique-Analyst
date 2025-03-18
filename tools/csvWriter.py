
import uuid

def write_csv(csv_text: str) -> str:
    """
    Use this function to create csv file from content.

    Args:
        csv_text (str): The csv content to save to file.
    
    Returns:
        str: The path to the csv file.
    """

    file_name = f"{uuid.uuid4()}.csv"
    with open(file_name, "w") as file:
        file.write(csv_text)
    print(f"CSV file created: {file_name}")
    return f"/?param={file_name}"

