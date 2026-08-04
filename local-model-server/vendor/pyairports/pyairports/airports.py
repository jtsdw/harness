"""Shim for the abandoned/empty `pyairports` 0.0.1 PyPI package.

`outlines.types.airports` does `from pyairports.airports import AIRPORT_LIST` and expects an
iterable of tuples where index 3 is the IATA code. The real `pyairports==0.0.1` wheel on PyPI
ships no code at all (metadata only), so anything importing it fails with ModuleNotFoundError.
This rebuilds an equivalent AIRPORT_LIST from `airportsdata`, which does ship real data.
"""

import airportsdata

_data = airportsdata.load("IATA")

AIRPORT_LIST = [
    (
        airport.get("name", ""),
        airport.get("city", ""),
        airport.get("country", ""),
        iata,
        airport.get("icao", ""),
        airport.get("lat", 0.0),
        airport.get("lon", 0.0),
    )
    for iata, airport in _data.items()
]
