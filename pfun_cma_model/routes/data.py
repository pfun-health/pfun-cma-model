"""
PFun CMA Model - Data API Routes
"""
from io import StringIO, BytesIO
from fastapi import APIRouter, Request, Response, HTTPException, status
from starlette.responses import StreamingResponse
import pandas as pd
import logging
from typing import Any, Literal, Optional
from dataclasses import dataclass, InitVar, field, MISSING

from pfun_cma_model.data import read_sample_data

router = APIRouter()


PFunDatasetMediaType = Literal["json", "text", "html", "octet-stream"]


@dataclass
class PFunDatasetResponseFormatter:
    """
    A formatter class for converting pandas DataFrame responses into multiple output formats.
    This class provides methods to serialize a pandas DataFrame into JSON, CSV (text),
    and HTML formats, making it suitable for returning dataset responses in different
    content types.

    :var data: The pandas DataFrame containing the dataset to be formatted.
    :type data: pd.DataFrame
    """
    data: pd.DataFrame
    
    def __post_init__(self):
        """Post-initialization to set up formatted output methods."""
        # Map of available formatted output methods
        self.output_format_map = {
            "aggregate": {
                "json": self.json,
                "text": self.text,
                "html": self.html
            },
            "streaming": {
                "json": self.json_stream,
                "text": self.text_stream,
                "octet-stream": self.octet_stream
            }
        }

    def json(self) -> str:
        """
        return the dataset as a JSON formatted string.
        """
        return self.data.to_json(orient='records')

    def text(self) -> str:
        """
        return the dataset as a CSV formatted string.
        
        :param self: instance of the class
        :return: CSV formatted string of the dataset
        :rtype: str
        """
        buf = StringIO()
        self.data.to_csv(path_or_buf=buf, encoding='utf-8')  # type:ignore
        buf.seek(0)
        return buf.getvalue()
    
    def text_stream(self) -> Any:
        """Yield the dataset as a CSV formatted string stream."""
        buf = StringIO()
        self.data.to_csv(path_or_buf=buf, encoding='utf-8', index=False)  # type:ignore
        buf.seek(0)
        for line in buf:
            yield line
            
    def json_stream(self) -> Any:
        """Yield the dataset as a JSON formatted string stream."""
        records = self.data.to_dict(orient='records')
        for record in records:
            yield f"{pd.io.json.dumps(record)}\n"
            
    async def octet_stream(self) -> Any:
        """Yield the dataset as a binary octet-stream."""
        buf = BytesIO()
        self.data.iterrows()

    def html(self) -> str:
        return self.data.to_html()


@dataclass
class PFunDatasetResponse:
    data: Optional[pd.DataFrame] = field(  # type: ignore
        default=MISSING, default_factory=read_sample_data)  # type: ignore
    pct0: float = 0.0
    nrows: InitVar[int] = 23
    nrows_given: bool | None = None
    media_type: PFunDatasetMediaType = "json"

    def __post_init__(self, nrows: int):
        """Post-initialization to parse nrows and data."""
        _, self.nrows_given = self._parse_nrows(nrows)
        self.data: pd.DataFrame = self._parse_data(
            self.data, self.pct0, nrows, self.nrows_given)

    @property
    def response(self) -> Response:
        """Generate a Response object with the dataset as JSON."""
        logging.debug("Downloading aggregate formatted data sample response generated with media_type=%s", self.media_type)
        return Response(
            content=PFunDatasetResponseFormatter,
            status_code=200,
            headers={"Content-Type": "application/{self.media_type}"}
        )
    
    @property
    def streaming_response(self) -> StreamingResponse:
        """Generate a streaming Response object with the dataset as JSON."""
        return StreamingResponse(
            content=self.formatted_output,
            media_type=f"application/{self.media_type}"
        )
    
    @classmethod
    def _parse_data(cls, data: pd.DataFrame | None, pct0: float, nrows: int, nrows_given: bool):
        """Parse and limit the dataset based on pct0, nrows and nrows_given."""
        # If no data provided, read the default sample dataset
        if data is None:
            data = read_sample_data(convert2json=False)  # type: ignore
        # ensure pd.DataFrame
        dataset = pd.DataFrame(data)
        logging.debug("Sample dataset loaded with %d rows.", len(dataset))

        # Calculate row0 from pct0
        if not (0.0 <= pct0 <= 1.0):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="pct0 must be between 0.0 and 1.0.",
            )

        num_rows_total = len(dataset)
        row0 = int(pct0 * num_rows_total)

        if nrows_given:
            # limit the dataset to the specified number of rows, with wrapping
            indices = [(row0 + i) % num_rows_total for i in range(nrows)]
            return dataset.iloc[indices]  # type: ignore
        else:
            # no nrows limit, return from row0 to end
            return dataset.iloc[row0:, :]  # type: ignore

    @classmethod
    def _parse_nrows(cls, nrows: int) -> tuple[int, bool]:
        """Parse and validate the nrows parameter for dataset retrieval.
        Args:
            nrows (int): The number of rows to return. If -1, return the full dataset.
        Returns:
            tuple: A tuple containing the validated nrows and a boolean indicating if nrows was given.
        """
        # Check if nrows is valid
        if nrows < -1:
            logging.error(
                "Invalid nrows value: %s. Must be -1 or greater.", nrows)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="nrows must be -1 (for full dataset) or a non-negative integer.",
            )
        if nrows == -1:
            nrows_given = False  # -1 means no limit, return full dataset
        else:
            nrows_given = True  # nrows is given, return only the first nrows
        logging.debug(
            "Received request for sample dataset with nrows=%s", nrows)
        logging.debug("(nrows_given) Was nrows_given? %s",
                      "'Yes.'" if nrows_given else "'No.'")
        return nrows, nrows_given


@router.get("/sample/download")
def get_sample_dataset(
    request: Request,  # type: ignore
    nrows: int = 23,
    media_type: PFunDatasetMediaType = "text"
):
    """(slow) Download the sample dataset with optional row limit.

    Args:
        request (Request): The FastAPI request object.
        nrows (int): The number of rows to return. If -1, return the full dataset.
        media_type (PFunDatasetMediaType): The return type expected of the response. 
    """
    # Read the sample dataset (data=None means use default sample data)
    dataset_response = PFunDatasetResponse(
        data=None, nrows=nrows, media_type=media_type)
    return dataset_response.response


@router.get("/sample/stream")
async def stream_sample_dataset(
    request: Request,  # type: ignore
    pct0: float = 0.5,
    nrows: int = 10,
    media_type: PFunDatasetMediaType = "text"
) -> StreamingResponse:
    """(fast) Stream the sample dataset with optional row limit.
    Args:
        request (Request): The FastAPI request object.
        pct0 (float): The relative location to start in the dataset [0.0, 1.0].
        nrows (int): The number of rows to include in the stream. If -1, stream the full dataset.
    """
    dataset_response = PFunDatasetResponse(
        data=None, pct0=pct0, nrows=nrows, media_type=media_type)
    # return the iterable (generating) streaming response
    return dataset_response.streaming_response
