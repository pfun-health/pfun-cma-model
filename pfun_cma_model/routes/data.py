"""
PFun CMA Model - Data API Routes
"""
from io import StringIO
from fastapi import APIRouter, Request, Response, HTTPException, status
from starlette.responses import StreamingResponse
import pandas as pd
import logging
from typing import Any, Literal, Optional
from dataclasses import dataclass, InitVar, field, MISSING

from pfun_cma_model.data import read_sample_data

router = APIRouter()


PFunDatasetMediaType = Literal["json", "text", "html"]


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

    def json(self) -> str:
        return self.data.to_json(orient='records')

    def text(self) -> str:
        buf = StringIO()
        self.data.to_csv(path_or_buf=buf)  # type:ignore
        buf.seek(0)
        return buf.getvalue()

    def html(self) -> str:
        return self.data.to_html()


@dataclass
class PFunDatasetResponse:
    data: Optional[pd.DataFrame] = field(
        default=MISSING, default_factory=read_sample_data)
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
    def streaming_response(self) -> StreamingResponse:
        """Generate a streaming Response object with the dataset as JSON."""
        return StreamingResponse(
            content=self._stream,
            media_type=f"application/{self.media_type}"
        )

    @property
    def _stream(self) -> Any:
        """Yield the dataset as streamable generator."""
        if self.media_type == 'json':
            yield '[\n'
        for record in self.formatted_output.split("\n"):
            yield record
        if self.media_type == 'json':
            yield ']'

    @property
    def formatted_output(self) -> str:
        """
        return the formatted output to be passed as content to the response.

        :param self: Description
        :return: Description
        :rtype: Any
        """
        data: pd.DataFrame = self.data  # type: ignore
        response_formatter = PFunDatasetResponseFormatter(data=data)
        output = getattr(response_formatter, self.media_type)()
        return output

    @property
    def response(self) -> Response:
        """Generate a Response object with the dataset as JSON."""
        formatted_output = self.formatted_output
        return Response(
            content=formatted_output,
            status_code=200,
            headers={"Content-Type": "application/{self.media_type}"}
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
    request: Request,
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
    request: Request,
    pct0: float = 0.0,
    nrows: int = -1,
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
