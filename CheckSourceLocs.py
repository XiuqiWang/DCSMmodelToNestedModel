#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Classify DCSM source/sink PLI points against a nested D-Flow FM mesh.

The script scans all ``*.pli`` files below SOURCE_SINK_DIRECTORY, converts
geographic source coordinates to EPSG:28992 when needed, and tests every point
against the actual child face polygons. It prints all locations and decisions,
then prints ready-to-copy PLI content for explicitly selected sources. It does
not move, copy, edit, or create any file.

Output classifications
----------------------
INCLUDE_CANDIDATE
    Point lies in a child face whose approximate bed elevation is below NAP.
    Still verify attachment and wetness in FlowFM.dia.

CHECK_CELL_WETNESS
    Point lies geometrically in a child face, but the approximate face bed
    elevation is non-negative or unavailable.

OUTSIDE_MESH_REVIEW_MOUTH_RELOCATION
    Point is outside the mesh but within the model's latitude range and close
    to it. Do not include it directly. Relocation to a represented river mouth
    requires an explicit modelling decision and documented coordinates.

DO_NOT_INCLUDE_OUTSIDE_MESH
    Point is outside the child mesh. Its influence should normally enter via
    DCSM-derived open-boundary conditions.

UNKNOWN_COORDINATE_SYSTEM
    Coordinate values are neither recognizable WGS84 nor RD New.

Dependencies
------------
conda install -c conda-forge xarray netcdf4 numpy scipy pyproj matplotlib
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
from matplotlib.path import Path as PolygonPath
from pyproj import Transformer
from scipy.spatial import cKDTree


# =============================================================================
# USER SETTINGS
# =============================================================================

SOURCE_SINK_DIRECTORY = Path(
    r"C:\dev\Python\DCSMmodelToNestedModel\Source_sinks"
)

CHILD_GRID_FILE = Path(
    r"C:\dev\Python\DCSMmodelToNestedModel\Nested_grid\flowgrid.nc"
)

# Sources that should be represented inside this particular child domain even
# though their original DCSM point lies just inland. Based on the domain review,
# IJMDBNN is included at the nearest eligible wet child face. MAASSS and
# HARVSZBNN remain outside and are represented through DCSM-derived boundaries.
SOURCES_TO_RELOCATE_AND_INCLUDE = {"IJMDBNN"}

# A source already inside an actual child face is printed as an inclusion
# candidate automatically. Its coordinate is transformed to RD New but is not
# moved to a face centre.
INCLUDE_SOURCES_ALREADY_INSIDE_MESH = True

# Outside points closer than this to a child face centre are flagged for a
# possible, manually justified mouth-location review. They are never included
# automatically.
MOUTH_REVIEW_DISTANCE_M = 20000.0

# Candidate polygons tested around each source point. The child mesh is nearly
# regular; 128 is deliberately conservative near irregular coastal faces.
NUMBER_OF_CANDIDATE_FACES = 128

# Approximate bed elevation used only for a warning/classification. Exact
# source attachment and initial wetness must still be checked in FlowFM.dia.
APPROXIMATE_WET_BED_THRESHOLD_M = 0.0


# =============================================================================
# PLI READING
# =============================================================================

def two_integers(fields: List[str]) -> Optional[Tuple[int, int]]:
    if len(fields) < 2:
        return None
    try:
        first = int(fields[0])
        second = int(fields[1])
    except ValueError:
        return None
    return first, second


def read_pli_blocks(pli_file: Path) -> List[Dict[str, object]]:
    """Read standard PLI blocks without mistaking dimension rows for points."""
    raw_lines = pli_file.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()
    lines = [
        line.strip() for line in raw_lines
        if line.strip() and not line.lstrip().startswith(('*', '#'))
    ]

    points = []  # type: List[Dict[str, object]]
    position = 0

    while position < len(lines):
        block_name = lines[position]
        position += 1
        if position >= len(lines):
            break

        dimensions = two_integers(lines[position].split())
        if dimensions is None:
            # Be tolerant of unexpected metadata lines and continue scanning.
            continue
        number_of_points, number_of_columns = dimensions
        position += 1

        if number_of_points < 1 or number_of_columns < 2:
            continue

        for point_number in range(number_of_points):
            if position >= len(lines):
                raise ValueError(
                    "Unexpected end of file in {} block {}".format(
                        pli_file.name, block_name
                    )
                )
            fields = lines[position].split()
            position += 1
            if len(fields) < 2:
                raise ValueError(
                    "Invalid coordinate row in {}: {}".format(
                        pli_file.name, lines[position - 1]
                    )
                )
            try:
                x = float(fields[0])
                y = float(fields[1])
            except ValueError:
                raise ValueError(
                    "Non-numeric PLI coordinates in {}: {}".format(
                        pli_file.name, lines[position - 1]
                    )
                )

            point_name = (
                fields[2]
                if len(fields) >= 3
                else "{}_{:04d}".format(block_name, point_number + 1)
            )
            points.append({
                "block_name": block_name,
                "point_number_one_based": point_number + 1,
                "point_name": point_name,
                "x_original": x,
                "y_original": y,
            })

    return points


# =============================================================================
# CHILD MESH
# =============================================================================

def finite_numeric(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    result[~np.isfinite(result)] = np.nan
    result[np.abs(result) > 1.0e30] = np.nan
    result[np.isclose(result, -999.0, rtol=0.0, atol=1.0e-10)] = np.nan
    return result


class ChildMesh(object):
    def __init__(self, grid_file: Path):
        with xr.open_dataset(
            grid_file, decode_times=False, mask_and_scale=False
        ) as ds:
            required = [
                "mesh2d_node_x", "mesh2d_node_y",
                "mesh2d_face_x", "mesh2d_face_y",
                "mesh2d_face_nodes",
            ]
            missing = [name for name in required if name not in ds]
            if missing:
                raise KeyError(
                    "Missing child-grid variable(s): {}".format(missing)
                )

            self.node_x = finite_numeric(ds["mesh2d_node_x"].values)
            self.node_y = finite_numeric(ds["mesh2d_node_y"].values)
            self.face_x = finite_numeric(ds["mesh2d_face_x"].values)
            self.face_y = finite_numeric(ds["mesh2d_face_y"].values)

            connectivity = ds["mesh2d_face_nodes"]
            self.face_nodes = np.asarray(connectivity.values)
            self.start_index = int(connectivity.attrs.get("start_index", 1))
            self.connectivity_fill = connectivity.attrs.get("_FillValue", -999)

            self.node_z = None
            if "mesh2d_node_z" in ds:
                self.node_z = finite_numeric(ds["mesh2d_node_z"].values)

        valid_faces = np.isfinite(self.face_x) & np.isfinite(self.face_y)
        if not np.all(valid_faces):
            raise ValueError("Child face centres contain invalid coordinates")

        self.tree = cKDTree(np.column_stack((self.face_x, self.face_y)))
        self.x_min = float(np.min(self.node_x[np.isfinite(self.node_x)]))
        self.x_max = float(np.max(self.node_x[np.isfinite(self.node_x)]))
        self.y_min = float(np.min(self.node_y[np.isfinite(self.node_y)]))
        self.y_max = float(np.max(self.node_y[np.isfinite(self.node_y)]))

        # Precompute an approximate bed level for selecting a receiving cell.
        # Face-centre coordinates are guaranteed to lie inside their own cells,
        # which makes them safer proposed PLI coordinates than an arbitrary
        # nearest point on a polygon edge.
        self.face_bed_level = np.full(self.face_x.size, np.nan, dtype=np.float64)
        if self.node_z is not None:
            for face_index in range(self.face_x.size):
                self.face_bed_level[face_index] = self.approximate_bed_level(
                    face_index
                )
        self.wet_face_indices = np.flatnonzero(
            np.isfinite(self.face_bed_level)
            & (self.face_bed_level < APPROXIMATE_WET_BED_THRESHOLD_M)
        )
        if self.wet_face_indices.size:
            self.wet_tree = cKDTree(np.column_stack((
                self.face_x[self.wet_face_indices],
                self.face_y[self.wet_face_indices],
            )))
        else:
            self.wet_tree = None

    def node_indices(self, face_index: int) -> np.ndarray:
        raw = np.asarray(self.face_nodes[face_index])
        valid = (
            np.isfinite(raw)
            & (raw != self.connectivity_fill)
            & (raw >= self.start_index)
        )
        return raw[valid].astype(np.int64) - self.start_index

    def polygon(self, face_index: int) -> np.ndarray:
        indices = self.node_indices(face_index)
        return np.column_stack((self.node_x[indices], self.node_y[indices]))

    def approximate_bed_level(self, face_index: int) -> float:
        if self.node_z is None:
            return np.nan
        values = self.node_z[self.node_indices(face_index)]
        values = values[np.isfinite(values)]
        return float(np.mean(values)) if values.size else np.nan

    def locate(self, x_rd: float, y_rd: float) -> Dict[str, object]:
        k = min(NUMBER_OF_CANDIDATE_FACES, self.face_x.size)
        distance, candidates = self.tree.query((x_rd, y_rd), k=k)
        distance = np.atleast_1d(distance)
        candidates = np.atleast_1d(candidates)

        containing_face = None  # type: Optional[int]
        for candidate in candidates:
            candidate = int(candidate)
            polygon = self.polygon(candidate)
            if polygon.shape[0] < 3:
                continue
            if PolygonPath(polygon, closed=True).contains_point(
                (x_rd, y_rd), radius=1.0e-7
            ):
                containing_face = candidate
                break

        nearest_wet_face = None
        nearest_wet_distance = np.nan
        proposed_x = np.nan
        proposed_y = np.nan
        proposed_bed = np.nan
        if self.wet_tree is not None:
            wet_distance, wet_local_index = self.wet_tree.query((x_rd, y_rd), k=1)
            nearest_wet_face = int(self.wet_face_indices[int(wet_local_index)])
            nearest_wet_distance = float(wet_distance)
            proposed_x = float(self.face_x[nearest_wet_face])
            proposed_y = float(self.face_y[nearest_wet_face])
            proposed_bed = float(self.face_bed_level[nearest_wet_face])

        return {
            "inside_mesh": containing_face is not None,
            "containing_face_zero_based": containing_face,
            "nearest_face_centre_distance_m": float(distance[0]),
            "approximate_face_bed_level_m": (
                self.approximate_bed_level(containing_face)
                if containing_face is not None else np.nan
            ),
            "inside_mesh_bounding_box": (
                self.x_min <= x_rd <= self.x_max
                and self.y_min <= y_rd <= self.y_max
            ),
            "nearest_wet_face_zero_based": nearest_wet_face,
            "nearest_wet_face_distance_m": nearest_wet_distance,
            "proposed_wet_face_x_rd_m": proposed_x,
            "proposed_wet_face_y_rd_m": proposed_y,
            "proposed_wet_face_bed_level_m": proposed_bed,
        }


# =============================================================================
# CLASSIFICATION
# =============================================================================

WGS84_TO_RD = Transformer.from_crs(
    "EPSG:4326", "EPSG:28992", always_xy=True
)


def coordinate_to_rd(x: float, y: float) -> Tuple[float, float, str]:
    if -180.0 <= x <= 180.0 and -90.0 <= y <= 90.0:
        x_rd, y_rd = WGS84_TO_RD.transform(x, y)
        return float(x_rd), float(y_rd), "EPSG:4326"

    # Broad but safe RD New range around the Netherlands.
    if -100000.0 <= x <= 400000.0 and 200000.0 <= y <= 700000.0:
        return float(x), float(y), "EPSG:28992_assumed"

    return np.nan, np.nan, "unknown"


def related_signal_files(pli_file: Path, point_name: str) -> List[str]:
    candidates = []  # type: List[Path]
    patterns = [
        point_name + ".tim",
        point_name + ".cmp",
        pli_file.stem + "*.tim",
        pli_file.stem + "*.cmp",
    ]
    for pattern in patterns:
        candidates.extend(pli_file.parent.glob(pattern))
    unique = sorted(set(path.resolve() for path in candidates if path.is_file()))
    return [path.name for path in unique]


def classify(
    inside_mesh: bool,
    approximate_bed: float,
    inside_latitude_range: bool,
    nearest_distance: float,
) -> str:
    if inside_mesh:
        if (
            np.isfinite(approximate_bed)
            and approximate_bed < APPROXIMATE_WET_BED_THRESHOLD_M
        ):
            return "INCLUDE_CANDIDATE"
        return "CHECK_CELL_WETNESS"

    if inside_latitude_range and nearest_distance <= MOUTH_REVIEW_DISTANCE_M:
        return "OUTSIDE_MESH_REVIEW_MOUTH_RELOCATION"
    return "DO_NOT_INCLUDE_OUTSIDE_MESH"


def main() -> None:
    if not SOURCE_SINK_DIRECTORY.is_dir():
        raise FileNotFoundError(
            "Source/sink directory not found: {}".format(SOURCE_SINK_DIRECTORY)
        )
    if not CHILD_GRID_FILE.is_file():
        raise FileNotFoundError(
            "Child grid not found: {}".format(CHILD_GRID_FILE)
        )

    pli_files = sorted(SOURCE_SINK_DIRECTORY.rglob("*.pli"))
    if not pli_files:
        raise FileNotFoundError(
            "No PLI files found below {}".format(SOURCE_SINK_DIRECTORY)
        )

    print("Found {} PLI files".format(len(pli_files)))
    mesh = ChildMesh(CHILD_GRID_FILE)

    # Transform the child y-range corners to geographic latitude. This range is
    # only a review flag; it never establishes inclusion.
    rd_to_wgs84 = Transformer.from_crs(
        "EPSG:28992", "EPSG:4326", always_xy=True
    )
    corner_x = np.array([mesh.x_min, mesh.x_min, mesh.x_max, mesh.x_max])
    corner_y = np.array([mesh.y_min, mesh.y_max, mesh.y_min, mesh.y_max])
    _, corner_lat = rd_to_wgs84.transform(corner_x, corner_y)
    latitude_min = float(np.min(corner_lat))
    latitude_max = float(np.max(corner_lat))

    records = []  # type: List[Dict[str, object]]
    parse_errors = []  # type: List[str]

    for number, pli_file in enumerate(pli_files, 1):
        print("[{0:03d}/{1:03d}] {2}".format(
            number, len(pli_files), pli_file.relative_to(SOURCE_SINK_DIRECTORY)
        ))
        try:
            points = read_pli_blocks(pli_file)
        except Exception as error:
            parse_errors.append("{}: {}".format(pli_file, error))
            continue

        if not points:
            parse_errors.append("{}: no coordinate records found".format(pli_file))
            continue

        for point in points:
            original_x = float(point["x_original"])
            original_y = float(point["y_original"])
            x_rd, y_rd, source_crs = coordinate_to_rd(original_x, original_y)

            base = {
                "relative_pli_path": str(pli_file.relative_to(SOURCE_SINK_DIRECTORY)),
                "pli_file": pli_file.name,
                "block_name": point["block_name"],
                "point_name": point["point_name"],
                "point_number_one_based": point["point_number_one_based"],
                "original_x": original_x,
                "original_y": original_y,
                "detected_source_crs": source_crs,
                "x_rd_m": x_rd,
                "y_rd_m": y_rd,
            }

            if source_crs == "unknown":
                base.update({
                    "inside_mesh": False,
                    "containing_face_zero_based": np.nan,
                    "nearest_face_centre_distance_m": np.nan,
                    "approximate_face_bed_level_m": np.nan,
                    "inside_mesh_bounding_box": False,
                    "inside_child_latitude_range": False,
                    "classification": "UNKNOWN_COORDINATE_SYSTEM",
                    "related_signal_files": "",
                    "signal_file_count": 0,
                    "decision": "EXCLUDE",
                    "decision_reason": "Unknown coordinate system",
                    "corrected_x_rd_m": np.nan,
                    "corrected_y_rd_m": np.nan,
                })
                records.append(base)
                continue

            location = mesh.locate(x_rd, y_rd)
            inside_latitude_range = (
                latitude_min <= original_y <= latitude_max
                if source_crs == "EPSG:4326"
                else mesh.y_min <= y_rd <= mesh.y_max
            )
            signals = related_signal_files(pli_file, str(point["point_name"]))
            action = classify(
                bool(location["inside_mesh"]),
                float(location["approximate_face_bed_level_m"]),
                inside_latitude_range,
                float(location["nearest_face_centre_distance_m"]),
            )
            base.update(location)
            base.update({
                "inside_child_latitude_range": inside_latitude_range,
                "classification": action,
                "related_signal_files": "; ".join(signals),
                "signal_file_count": len(signals),
            })

            source_id = str(point["block_name"]).strip().upper()
            if bool(location["inside_mesh"]) and INCLUDE_SOURCES_ALREADY_INSIDE_MESH:
                base.update({
                    "decision": "INCLUDE_AT_ORIGINAL_LOCATION",
                    "decision_reason": "Original point lies inside a child face",
                    "corrected_x_rd_m": x_rd,
                    "corrected_y_rd_m": y_rd,
                })
            elif source_id in SOURCES_TO_RELOCATE_AND_INCLUDE:
                if location["nearest_wet_face_zero_based"] is None:
                    raise RuntimeError(
                        "Cannot propose a wet receiving cell for {} because "
                        "no face with bed level below the configured threshold "
                        "was found.".format(source_id)
                    )
                base.update({
                    "decision": "INCLUDE_AT_PROPOSED_WET_FACE",
                    "decision_reason": (
                        "Selected coastal source; original point is inland/outside "
                        "the mesh"
                    ),
                    "corrected_x_rd_m": location["proposed_wet_face_x_rd_m"],
                    "corrected_y_rd_m": location["proposed_wet_face_y_rd_m"],
                })
            else:
                base.update({
                    "decision": "EXCLUDE",
                    "decision_reason": (
                        "Source lies outside this child domain; retain its "
                        "influence through DCSM-derived open boundaries"
                    ),
                    "corrected_x_rd_m": np.nan,
                    "corrected_y_rd_m": np.nan,
                })
            records.append(base)

    if not records:
        raise RuntimeError("No usable source/sink point records were found")

    records.sort(key=lambda row: (
        0 if str(row["decision"]).startswith("INCLUDE") else 1,
        str(row["relative_pli_path"]),
        int(row["point_number_one_based"]),
    ))

    print("\n" + "=" * 100)
    print("ALL SOURCE LOCATIONS AND DECISIONS")
    print("=" * 100)
    for row in records:
        print("\nSource       : {}".format(row["block_name"]))
        print("PLI file     : {}".format(row["relative_pli_path"]))
        print("Point name   : {}".format(row["point_name"]))
        print("Original     : {:.9f}, {:.9f} ({})".format(
            float(row["original_x"]), float(row["original_y"]),
            row["detected_source_crs"],
        ))
        if np.isfinite(float(row["x_rd_m"])):
            print("Original RD  : {:.3f}, {:.3f}".format(
                float(row["x_rd_m"]), float(row["y_rd_m"])
            ))
            print("Inside mesh  : {}".format(row["inside_mesh"]))
            print("Nearest face : {:.1f} m".format(
                float(row["nearest_face_centre_distance_m"])
            ))
            print("Nearest wet  : {:.1f} m, face {}".format(
                float(row["nearest_wet_face_distance_m"]),
                row["nearest_wet_face_zero_based"],
            ))
        print("Signals      : {}".format(
            row["related_signal_files"] or "NONE FOUND"
        ))
        print("Decision     : {}".format(row["decision"]))
        print("Reason       : {}".format(row["decision_reason"]))

    included = [
        row for row in records
        if str(row["decision"]).startswith("INCLUDE")
    ]
    print("\n" + "=" * 100)
    print("SOURCES TO INCLUDE")
    print("=" * 100)
    if not included:
        print("None")
    for row in included:
        print("\n{}: {}".format(row["block_name"], row["decision"]))
        print("Corrected EPSG:28992 coordinate: {:.3f}  {:.3f}".format(
            float(row["corrected_x_rd_m"]),
            float(row["corrected_y_rd_m"]),
        ))
        print("Associated signal files: {}".format(
            row["related_signal_files"] or "NONE FOUND"
        ))
        print("Corrected PLI content:")
        print(str(row["block_name"]))
        print("    1    3")
        print("{:.3f}  {:.3f}  {}".format(
            float(row["corrected_x_rd_m"]),
            float(row["corrected_y_rd_m"]),
            row["point_name"],
        ))

    if parse_errors:
        print("\nPLI files requiring manual inspection")
        for error in parse_errors:
            print("  " + error)

    print(
        "\nNo files were written. Proposed face-centre coordinates must be "
        "checked visually and confirmed in FlowFM.dia before production."
    )


if __name__ == "__main__":
    main()
