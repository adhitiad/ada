import pandas as pd
from langchain.schema import Document
from langchain_community.document_loaders.csv_loader import CSVLoader


class CustomCSVLoader(CSVLoader):
    """
    Custom CSV loader that handles potential errors and allows specifying the content column.
    """

    def __init__(
        self,
        file_path,
        content_column="text",
        encoding: str = "auto",
        **kwargs,
    ):
        """
        Initializes the CSV loader.

        Args:
            file_path: Path to the CSV file.
            content_column: The name of the column to use as document content. Defaults to "text".
            encoding: File encoding to use. Default "auto" will try different encodings.
            **kwargs: Additional keyword arguments to pass to the CSVLoader.
        """
        super().__init__(file_path, **kwargs)
        self.content_column = content_column
        self.encoding = encoding

    def load(self) -> list[Document]:
        """Loads data from the CSV file. Handles potential errors gracefully."""
        try:
            if self.encoding == "auto":
                encodings = ["utf-8", "latin1", "iso-8859-1", "cp1252"]
                for enc in encodings:
                    try:
                        df = pd.read_csv(self.file_path, encoding=enc)
                        break
                    except UnicodeDecodeError:
                        continue
            else:
                df = pd.read_csv(self.file_path, encoding=self.encoding)

            if self.content_column not in df.columns:
                raise ValueError(
                    f"Column '{self.content_column}' not found in CSV file."
                )

            loaded_documents = []
            for index, row in df.iterrows():
                content = str(
                    row[self.content_column]
                )  # Handle potential non-string values
                metadata = {"row_index": index}
                metadata.update(row.to_dict())  # Add all columns as metadata

                loaded_documents.append(
                    Document(page_content=content, metadata=metadata)
                )
            return loaded_documents
        except FileNotFoundError:
            print(f"Error: File not found at {self.file_path}")
            return []
        except pd.errors.EmptyDataError:
            print(f"Error: CSV file at {self.file_path} is empty.")
            return []
        except pd.errors.ParserError:
            print(
                f"Error: Could not parse CSV file at {self.file_path}. Check its format."
            )
            return []
        except ValueError as e:
            print(f"Error: {e}")

            return []
