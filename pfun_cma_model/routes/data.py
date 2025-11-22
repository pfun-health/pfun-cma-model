"""
PFun CMA Model - Data API Routes
"""
from fastapi import APIRouter, Request, Response, HTTPException, status
from starlette.responses import StreamingResponse
from pandas import DataFrame
import logging
import json
from typing import Any
from dataclasses import dataclass, InitVar

from pfun_cma_model.data import read_sample_data

router = APIRouter()


@dataclass
class PFunDatasetResponse:
    data: DataFrame | None = None
    pct0: float = 0.0
    nrows: InitVar[int] = 23
    nrows_given: bool | None = None

    def __post_init__(self, nrows: int):
        """Post-initialization to parse nrows and data."""
        _, self.nrows_given = self._parse_nrows(nrows)
        self.data = self._parse_data(
            self.data, self.pct0, nrows, self.nrows_given)

    @property
    def streaming_response(self) -> StreamingResponse:
        """Generate a streaming Response object with the dataset as JSON."""
        return StreamingResponse(
            content=self._stream,
            media_type="application/json"
        )

    @property
    def response(self) -> Response:
        """Generate a Response object with the dataset as JSON."""
        output = self.data.to_json(orient='records')  # type: ignore
        return Response(
            content=output,
            status_code=200,
            headers={"Content-Type": "application/json"}
        )

    @classmethod
    def _parse_data(cls, data: DataFrame | None, pct0: float, nrows: int, nrows_given: bool):
        """Parse and limit the dataset based on pct0, nrows and nrows_given."""
        # If no data provided, read the default sample dataset
        if data is None:
            data = read_sample_data(convert2json=False)  # type: ignore
        # ensure DataFrame
        dataset = DataFrame(data)
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

    @property
    def _stream(self) -> Any:
        """Yield the dataset as streamable chunks."""
        rec_array = self.data.to_dict(orient='records')  # type: ignore
        for record in rec_array:  # type: ignore
            yield json.dumps(record) + '\n'

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
def get_sample_dataset(request: Request, nrows: int = 23):
    """(slow) Download the sample dataset with optional row limit.

    Args:
        request (Request): The FastAPI request object.
        nrows (int): The number of rows to return. If -1, return the full dataset.
    """
    # Read the sample dataset (data=None means use default sample data)
    dataset_response = PFunDatasetResponse(data=None, nrows=nrows)
    return dataset_response.response


@router.get("/sample/stream")
async def stream_sample_dataset(request: Request, pct0: float = 0.0, nrows: int = -1) -> StreamingResponse:
    """(fast) Stream the sample dataset with optional row limit.
    Args:
        request (Request): The FastAPI request object.
        pct0 (float): The relative location to start in the dataset [0.0, 1.0].
        nrows (int): The number of rows to include in the stream. If -1, stream the full dataset.
    """
    dataset_response = PFunDatasetResponse(
        data=None, pct0=pct0, nrows=nrows)
    # return the iterable (generating) streaming response
    return dataset_response.streaming_response