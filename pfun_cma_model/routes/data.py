"""
PFun CMA Model - Data API Routes
"""
import logging
from io import BytesIO, StringIO
from typing import Literal

import pandas as pd
from fastapi import APIRouter, HTTPException, Request, Response, status
from starlette.responses import StreamingResponse

from pfun_cma_model.data import read_sample_data

router = APIRouter()


PFunDatasetMediaType = Literal["json", "text", "html", "octet-stream"]


def _parse_nrows(nrows: int) -> tuple[int, bool]:
    """Parse and validate the nrows parameter for dataset retrieval.
    Args:
        nrows (int): nr rows to return. if -1, return the full dataset.
    Returns:
        tuple: A tuple containing the validated nrows (nrows was given)
    """
    # Check if nrows is valid
    if nrows < -1:
        logging.error(
            "Invalid nrows value: %s. Must be -1 or greater.", nrows)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="nrows must be -1 (for full dataset) or a non-negative integer.",  # noqa
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


def parse_data_parameters(
        data: pd.DataFrame | None, pct0: float, nrows: int
) -> pd.DataFrame | pd.Series:
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

    # Determine nrows and what the value means
    _, nrows_given = _parse_nrows(nrows=nrows)

    # Limit the dataset based on nrows and nrows_given
    if not nrows_given:
        # no nrows limit, return from row0 to end
        return dataset.iloc[row0:, :]  # type: ignore
    # limit the dataset to the specified number of rows, with wrapping
    indices = [(row0 + i) % num_rows_total for i in range(nrows)]
    return dataset.iloc[indices]  # type: ignore


@router.get("/sample/download")
def get_sample_dataset(
    request: Request,  # type: ignore
    nrows: int = 23,
    media_type: PFunDatasetMediaType = "html"
) -> Response:
    """Download the sample dataset with optional row limit.

    Args:
        request (Request): The FastAPI request object.
        nrows (int): The number of rows to return. If -1, return the full dataset.  # noqa
        media_type (PFunDatasetMediaType): The return type expected of the response.   # noqa
    """
    # Read the sample dataset (data=None means use default sample data)
    dataset = parse_data_parameters(
        data=None, pct0=0.0, nrows=nrows)
    # Prepare the response based on media_type
    if media_type == "json":
        return Response(
            content=dataset.to_json(orient="records"),
            media_type="application/json",
        )
    elif media_type == "text":
        return Response(
            content=dataset.to_csv(index=False),
            media_type="text/csv",
        )
    elif media_type == "html":
        return Response(
            content=dataset.to_html(index=False),
            media_type="text/html",
        )
    elif media_type == "octet-stream":
        return Response(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            content="Octet-stream download not implemented in non-streaming endpoint.",  # type: ignore
        )


@router.get("/sample/stream")
async def stream_sample_dataset(
    request: Request,  # type: ignore
    pct0: float = 0.5,
    nrows: int = 10,
    media_type: PFunDatasetMediaType = "text"
) -> StreamingResponse:
    """(streaming-optimized) Stream the sample dataset with optional row limit.
    Args:
        request (Request): The FastAPI request object.
        pct0 (float): The relative location to start in the dataset [0.0, 1.0].
        nrows (int): The number of rows to include in the stream. If -1, stream the full dataset.  # noqa
    """
    # Read the sample dataset (data=None means use default sample data)
    dataset = parse_data_parameters(
        data=None, pct0=pct0, nrows=nrows)

    # Prepare the response based on media_type
    if media_type == "json":
        stream = StringIO(dataset.to_json(orient="records"))
        return StreamingResponse(
            content=stream,
            media_type="application/json",
        )
    elif media_type == "text":
        stream = StringIO(dataset.to_csv(index=False))
        return StreamingResponse(
            content=stream,
            media_type="text/csv",
        )
    elif media_type == "html":
        stream = StringIO(dataset.to_html(index=False))
        return StreamingResponse(
            content=stream,
            media_type="text/html",
        )
    elif media_type == "octet-stream":
        logging.debug("streaming response with media_type=%s", media_type)
        buffer = BytesIO()
        # the client expects the extra index to be removed, header should be gone also
        dataset.to_csv(
            buffer, index=False, header=False
        )
        buffer.seek(0)
        return StreamingResponse(
            content=buffer,
            media_type="application/octet-stream",
            headers={
                'Content-Type': 'application/octet-stream',
                'Transfer-Encoding': 'chunked',
            }
        )
