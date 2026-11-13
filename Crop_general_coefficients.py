# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 16:16:51 2026

@author: WangX3
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prepare DCSM friction/mixing/Secchi spatial forcing for a child model.

Inputs are read from ``General`` in their original DCSM coordinate system.
The child mesh is read from ``Nested_grid/flowgrid.nc`` (EPSG:28992).

The script:
  * constructs the child model footprint from all mesh faces;
  * writes that footprint as ``nested_model_domain.pol``;
  * transforms and crops the friction-correction XYZ with a buffer so Delaunay
    interpolation remains defined at child boundaries;
  * transforms, intersects, and writes only viscosity/diffusivity/Secchi
    override polygons that overlap the child mesh;
  * writes an old-style EXT snippet containing only relevant blocks.

Original files are never edited.

Dependencies
------------
conda install -c conda-forge xarray netcdf4 numpy pyproj shapely
"""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import math

import numpy as np
import xarray as xr
from pyproj import Transformer
from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon
from shapely.ops import transform as transform_geometry
from shapely.ops import unary_union
from shapely.prepared import prep


# =============================================================================
# USER SETTINGS
# =============================================================================

PROJECT_DIRECTORY = Path(r"C:\dev\Python\DCSMmodelToNestedModel")
GENERAL_DIRECTORY = PROJECT_DIRECTORY / "General"
CHILD_GRID_FILE = PROJECT_DIRECTORY / "Nested_grid" / "flowgrid.nc"

OUTPUT_DIRECTORY = PROJECT_DIRECTORY / "Nested_general"

# The generated files should eventually be copied to this folder relative to
# FlowFM.mdu. Change this prefix if your final model layout is different.
EXT_FILE_PREFIX = "general"

# Retain friction XYZ samples around, not just inside, the child mesh. FILETYPE
# 7 uses triangulation; surrounding points are needed to avoid an undefined
# interpolation strip along the boundary.
XYZ_BUFFER_M = 20000.0

# Build the mesh union in batches to limit peak geometry overhead.
UNION_BATCH_SIZE = 2000

# Expected DCSM files.
FRICTION_XYZ = "frictioncorrection_dcsm-fm_0_5nm_v2_A05_work67.xyz"

POLYGON_OVERRIDES = [
    # input filename, output filename, coefficient value, role
    (
        "DCSM-FM_vicouv_20200124_BalticBnd.pol",
        "nested_vicouv_BalticBnd.pol",
        2000.0,
        "mixing",
    ),
    (
        "DCSM-FM_vicouv_inner_20200210.pol",
        "nested_vicouv_inner.pol",
        500.0,
        "mixing",
    ),
    (
        "DCSM-FM_vicouv_inner2_20200210.pol",
        "nested_vicouv_inner2.pol",
        100.0,
        "mixing",
    ),
    (
        "waddensea.pol",
        "nested_waddensea.pol",
        1.0,
        "secchi",
    ),
]


# =============================================================================
# GENERAL HELPERS
# =============================================================================

WGS84_TO_RD = Transformer.from_crs(
    "EPSG:4326", "EPSG:28992", always_xy=True
)


def finite_numeric(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    result[~np.isfinite(result)] = np.nan
    result[np.abs(result) > 1.0e30] = np.nan
    result[np.isclose(result, -999.0, rtol=0.0, atol=1.0e-10)] = np.nan
    return result


def geometry_parts(geometry) -> Iterable[Polygon]:
    """Yield non-empty polygon components from an arbitrary intersection."""
    if geometry is None or geometry.is_empty:
        return
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        for part in geometry.geoms:
            if not part.is_empty:
                yield part
    elif isinstance(geometry, GeometryCollection):
        for item in geometry.geoms:
            for part in geometry_parts(item):
                yield part


def clean_polygon(polygon: Polygon) -> Optional[Polygon]:
    if polygon.is_empty:
        return None
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty:
        return None
    if isinstance(polygon, Polygon):
        return polygon
    return None


def detect_geographic(x: np.ndarray, y: np.ndarray) -> bool:
    valid = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        raise ValueError("No valid coordinates available for CRS detection")
    return bool(
        np.nanmax(np.abs(x[valid])) <= 180.0
        and np.nanmax(np.abs(y[valid])) <= 90.0
    )


def transform_xy_to_rd(
    x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, str]:
    if detect_geographic(x, y):
        x_rd, y_rd = WGS84_TO_RD.transform(x, y)
        return (
            np.asarray(x_rd, dtype=np.float64),
            np.asarray(y_rd, dtype=np.float64),
            "EPSG:4326 -> EPSG:28992",
        )
    return x.astype(np.float64), y.astype(np.float64), "already projected"


# =============================================================================
# CHILD DOMAIN FROM THE GRID
# =============================================================================

def read_child_domain(grid_file: Path):
    print("Reading child grid: {}".format(grid_file))
    with xr.open_dataset(
        grid_file, decode_times=False, mask_and_scale=False
    ) as ds:
        required = ["mesh2d_node_x", "mesh2d_node_y", "mesh2d_face_nodes"]
        missing = [name for name in required if name not in ds]
        if missing:
            raise KeyError("Missing child-grid variables: {}".format(missing))

        node_x = finite_numeric(ds["mesh2d_node_x"].values)
        node_y = finite_numeric(ds["mesh2d_node_y"].values)
        connectivity = ds["mesh2d_face_nodes"]
        face_nodes = np.asarray(connectivity.values)
        start_index = int(connectivity.attrs.get("start_index", 1))
        fill_value = connectivity.attrs.get("_FillValue", -999)

    face_polygons = []  # type: List[Polygon]
    for face in range(face_nodes.shape[0]):
        raw = np.asarray(face_nodes[face])
        valid = (
            np.isfinite(raw)
            & (raw != fill_value)
            & (raw >= start_index)
        )
        indices = raw[valid].astype(np.int64) - start_index
        if indices.size < 3:
            continue
        coordinates = np.column_stack((node_x[indices], node_y[indices]))
        if not np.all(np.isfinite(coordinates)):
            continue
        polygon = clean_polygon(Polygon(coordinates))
        if polygon is not None and polygon.area > 0.0:
            face_polygons.append(polygon)

    if not face_polygons:
        raise RuntimeError("No valid child face polygons could be constructed")

    print("Constructed {:,} child face polygons".format(len(face_polygons)))
    partial_unions = []
    for start in range(0, len(face_polygons), UNION_BATCH_SIZE):
        stop = min(start + UNION_BATCH_SIZE, len(face_polygons))
        print("  Union faces {:,}-{:,.0f}".format(start + 1, stop))
        partial_unions.append(unary_union(face_polygons[start:stop]))
    domain = unary_union(partial_unions)
    if not domain.is_valid:
        domain = domain.buffer(0)
    if domain.is_empty:
        raise RuntimeError("Child-domain union is empty")
    print("Child footprint area: {:.3f} km2".format(domain.area / 1.0e6))
    return domain


# =============================================================================
# POLYGON FILES
# =============================================================================

def parse_pol_file(path: Path) -> List[Tuple[str, Polygon]]:
    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    lines = [
        line.strip() for line in raw_lines
        if line.strip() and not line.lstrip().startswith(("*", "#"))
    ]
    blocks = []  # type: List[Tuple[str, Polygon]]
    position = 0
    while position < len(lines):
        name = lines[position]
        position += 1
        if position >= len(lines):
            break
        dimensions = lines[position].split()
        position += 1
        if len(dimensions) < 2:
            raise ValueError("Invalid dimensions in {} block {}".format(path, name))
        try:
            number_of_points = int(dimensions[0])
            number_of_columns = int(dimensions[1])
        except ValueError:
            raise ValueError("Invalid dimensions in {} block {}".format(path, name))
        if number_of_points < 3 or number_of_columns < 2:
            raise ValueError("Invalid polygon size in {} block {}".format(path, name))

        coordinates = []
        for _ in range(number_of_points):
            if position >= len(lines):
                raise ValueError("Unexpected end of {} block {}".format(path, name))
            fields = lines[position].split()
            position += 1
            if len(fields) < 2:
                raise ValueError("Invalid coordinate row in {}".format(path))
            coordinates.append((float(fields[0]), float(fields[1])))

        array = np.asarray(coordinates, dtype=np.float64)
        x_rd, y_rd, transformation = transform_xy_to_rd(array[:, 0], array[:, 1])
        polygon = clean_polygon(Polygon(np.column_stack((x_rd, y_rd))))
        if polygon is not None:
            blocks.append((name, polygon))

    return blocks


def write_pol_file(path: Path, named_polygons: Sequence[Tuple[str, Polygon]]) -> int:
    block_number = 0
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for original_name, geometry in named_polygons:
            for polygon in geometry_parts(geometry):
                if polygon.area <= 0.01:
                    continue
                block_number += 1
                coordinates = np.asarray(polygon.exterior.coords, dtype=np.float64)
                # D-Flow polygon readers accept a closed ring; retain it.
                name = "{}_{:03d}".format(original_name, block_number)
                output.write(name + "\n")
                output.write("{} 2\n".format(coordinates.shape[0]))
                for x_value, y_value in coordinates:
                    output.write("{:.3f}  {:.3f}\n".format(x_value, y_value))
    return block_number


def crop_polygon_file(
    input_path: Path,
    output_path: Path,
    child_domain,
) -> int:
    blocks = parse_pol_file(input_path)
    clipped = []  # type: List[Tuple[str, Polygon]]
    for name, polygon in blocks:
        if not polygon.intersects(child_domain):
            continue
        intersection = polygon.intersection(child_domain)
        for number, part in enumerate(geometry_parts(intersection), 1):
            clipped.append(("{}_clip{:03d}".format(name, number), part))
    if not clipped:
        return 0
    return write_pol_file(output_path, clipped)


# =============================================================================
# FRICTION XYZ
# =============================================================================

def read_xyz(path: Path) -> np.ndarray:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith(("*", "#")):
            continue
        fields = stripped.split()
        if len(fields) < 3:
            continue
        try:
            rows.append((float(fields[0]), float(fields[1]), float(fields[2])))
        except ValueError:
            # Permit descriptive headers, but not malformed numeric rows.
            continue
    if not rows:
        raise ValueError("No XYZ samples found in {}".format(path))
    return np.asarray(rows, dtype=np.float64)


def crop_friction_xyz(
    input_path: Path,
    output_path: Path,
    child_domain,
) -> Tuple[int, int, str]:
    source = read_xyz(input_path)
    x_rd, y_rd, transformation = transform_xy_to_rd(source[:, 0], source[:, 1])
    values = source[:, 2]

    buffered = child_domain.buffer(XYZ_BUFFER_M)
    min_x, min_y, max_x, max_y = buffered.bounds
    bbox = (
        (x_rd >= min_x) & (x_rd <= max_x)
        & (y_rd >= min_y) & (y_rd <= max_y)
        & np.isfinite(values)
    )
    candidates = np.flatnonzero(bbox)
    prepared = prep(buffered)
    keep = [
        index for index in candidates
        if prepared.covers(Point(float(x_rd[index]), float(y_rd[index])))
    ]
    keep = np.asarray(keep, dtype=np.int64)
    if keep.size < 3:
        raise RuntimeError(
            "Only {} friction points remain after cropping; triangulation "
            "requires at least three.".format(keep.size)
        )

    output_data = np.column_stack((x_rd[keep], y_rd[keep], values[keep]))
    np.savetxt(
        output_path,
        output_data,
        fmt=["%.3f", "%.3f", "%.10g"],
        delimiter="  ",
    )
    return source.shape[0], keep.size, transformation


# =============================================================================
# EXT OUTPUT
# =============================================================================

def ext_path(filename: str) -> str:
    prefix = EXT_FILE_PREFIX.strip("/\\")
    return (prefix + "/" + filename) if prefix else filename


def write_ext_snippet(
    path: Path,
    friction_filename: str,
    included_overrides: List[Dict[str, object]],
) -> None:
    blocks = []

    def add_block(quantity, filename, filetype, method, operand, value=None):
        lines = [
            "QUANTITY     = {}".format(quantity),
            "FILENAME     = {}".format(ext_path(filename)),
            "FILETYPE     = {}".format(filetype),
            "METHOD       = {}".format(method),
            "OPERAND      = {}".format(operand),
        ]
        if value is not None:
            lines.append("VALUE        = {}".format(value))
        blocks.append("\n".join(lines))

    # The child-domain polygon replaces the DCSM-wide domain polygon.
    add_block("frictioncoefficient", "nested_model_domain.pol", 10, 4, "O", "0.028")
    add_block("frictioncoefficient", friction_filename, 7, 5, "*")

    add_block(
        "horizontaleddyviscositycoefficient",
        "nested_model_domain.pol", 10, 4, "O", "0.1",
    )
    for item in included_overrides:
        if item["role"] == "mixing":
            add_block(
                "horizontaleddyviscositycoefficient",
                str(item["output_name"]), 10, 4, "O", str(item["value"]),
            )

    add_block(
        "horizontaleddydiffusivitycoefficient",
        "nested_model_domain.pol", 10, 4, "O", "0.1",
    )
    for item in included_overrides:
        if item["role"] == "mixing":
            add_block(
                "horizontaleddydiffusivitycoefficient",
                str(item["output_name"]), 10, 4, "O", str(item["value"]),
            )

    add_block("secchidepth", "nested_model_domain.pol", 10, 4, "O", "4.0")
    for item in included_overrides:
        if item["role"] == "secchi":
            add_block(
                "secchidepth",
                str(item["output_name"]), 10, 4, "O", str(item["value"]),
            )

    header = (
        "# DCSM spatial coefficients transformed/cropped for the nested grid.\n"
        "# Merge these blocks into the OLD-format FlowFM external forcing file.\n"
        "# Generated paths assume the files are copied to dflowfm/{}.\n\n"
    ).format(EXT_FILE_PREFIX.strip("/\\"))
    path.write_text(header + "\n\n".join(blocks) + "\n", encoding="utf-8")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    if not GENERAL_DIRECTORY.is_dir():
        raise FileNotFoundError("General directory not found: {}".format(GENERAL_DIRECTORY))
    if not CHILD_GRID_FILE.is_file():
        raise FileNotFoundError("Child grid not found: {}".format(CHILD_GRID_FILE))

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    child_domain = read_child_domain(CHILD_GRID_FILE)

    domain_output = OUTPUT_DIRECTORY / "nested_model_domain.pol"
    domain_blocks = [
        ("nested_model_domain", polygon)
        for polygon in geometry_parts(child_domain)
    ]
    written_domain_blocks = write_pol_file(domain_output, domain_blocks)
    if written_domain_blocks < 1:
        raise RuntimeError("Failed to write child-domain polygon")
    print("Wrote {} domain block(s): {}".format(
        written_domain_blocks, domain_output.name
    ))

    friction_input = GENERAL_DIRECTORY / FRICTION_XYZ
    if not friction_input.is_file():
        raise FileNotFoundError("Friction correction not found: {}".format(friction_input))
    friction_output_name = "frictioncorrection_child_rd.xyz"
    friction_output = OUTPUT_DIRECTORY / friction_output_name
    original_count, cropped_count, transformation = crop_friction_xyz(
        friction_input, friction_output, child_domain
    )
    print(
        "Friction XYZ: {:,} -> {:,} points ({}), buffer={} km".format(
            original_count, cropped_count, transformation, XYZ_BUFFER_M / 1000.0
        )
    )

    included_overrides = []  # type: List[Dict[str, object]]
    print("\nPolygon intersections")
    for input_name, output_name, value, role in POLYGON_OVERRIDES:
        input_path = GENERAL_DIRECTORY / input_name
        output_path = OUTPUT_DIRECTORY / output_name
        if not input_path.is_file():
            print("  MISSING (skipped): {}".format(input_name))
            continue
        block_count = crop_polygon_file(input_path, output_path, child_domain)
        if block_count == 0:
            print("  NO INTERSECTION: {}".format(input_name))
            continue
        print("  INCLUDED: {} -> {} ({} block(s))".format(
            input_name, output_name, block_count
        ))
        included_overrides.append({
            "input_name": input_name,
            "output_name": output_name,
            "value": value,
            "role": role,
            "block_count": block_count,
        })

    ext_output = OUTPUT_DIRECTORY / "FlowFM_spatial_coefficients.ext"
    write_ext_snippet(
        ext_output, friction_output_name, included_overrides
    )

    print("\nCreated files")
    for output_file in sorted(OUTPUT_DIRECTORY.iterdir()):
        if output_file.is_file():
            print("  {}".format(output_file.name))
    print("\nEXT snippet: {}".format(ext_output))
    print(
        "Copy the generated spatial files into dflowfm/{}, then merge the "
        "snippet blocks into FlowFM.ext in the printed order.".format(
            EXT_FILE_PREFIX.strip("/\\")
        )
    )


if __name__ == "__main__":
    main()
