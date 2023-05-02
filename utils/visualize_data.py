import os
import re
from pathlib import Path
from argparse import ArgumentParser

import cv2
from typing import Optional, Tuple, Final, Any

import numpy as np
import numpy.typing as npt
import rerun as rr
from read_write_colmap import read_model, Camera

FILTER_MIN_VISIBLE: Final = 500


def scale_camera(camera: Camera, resize: Tuple[int, int]) -> Tuple[Camera, npt.NDArray[np.float_]]:
    """Scale the camera intrinsics to match the resized image."""
    assert camera.model == "PINHOLE" or camera.model == "OPENCV"
    new_width = resize[0]
    new_height = resize[1]
    scale_factor = np.array([new_width / camera.width, new_height / camera.height])

    # For PINHOLE camera model, params are: [focal_length_x, focal_length_y, principal_point_x, principal_point_y]
    new_params = np.append(camera.params[:2] * scale_factor, camera.params[2:4] * scale_factor)

    return (Camera(camera.id, camera.model, new_width, new_height, new_params), scale_factor)


def intrinsics_for_camera(camera: Camera) -> npt.NDArray[Any]:
    """Convert a colmap camera to a pinhole camera intrinsics matrix."""
    assert camera.model == "PINHOLE" or camera.model == "OPENCV"
    return np.vstack(
        [
            np.hstack(
                [
                    # Focal length is in [:2]
                    np.diag(camera.params[:2]),
                    # Principle point is in [2:]
                    np.vstack(camera.params[2:4]),
                ]
            ),
            [0, 0, 1],
        ]
    )


def read_and_log_sparse_reconstruction(
    dataset_path: Path, filter_output: bool, resize: Optional[Tuple[int, int]] = None
) -> None:
    print("Reading sparse COLMAP reconstruction")
    cameras, images, points3D = read_model(dataset_path / "sparse" / "0", ext=".bin")
    print("Building visualization by logging to Rerun")

    if filter_output:
        # Filter out noisy points
        points3D = {
            id: point
            for id, point in points3D.items()
            if point.rgb.any() and len(point.image_ids) > 4
        }

    rr.log_view_coordinates("/", up="-Y", timeless=True)

    # Iterate through images (video frames) logging data related to each frame.
    for image in sorted(images.values(), key=lambda im: im.name):  # type: ignore[no-any-return]
        image_file = dataset_path.parent / "images" / image.name

        if not os.path.exists(image_file):
            continue

        # COLMAP sets image ids that don't match the original video frame
        idx_match = re.search(r"\d+", image.name)
        assert idx_match is not None
        frame_idx = int(idx_match.group(0))

        quat_xyzw = image.qvec[[1, 2, 3, 0]]  # COLMAP uses wxyz quaternions
        camera_from_world = (
            image.tvec,
            quat_xyzw,
        )  # COLMAP's camera transform is "camera from world"
        camera = cameras[image.camera_id]
        if resize:
            camera, scale_factor = scale_camera(camera, resize)
        else:
            scale_factor = np.array([1.0, 1.0])

        intrinsics = intrinsics_for_camera(camera)

        visible = [
            id != -1 and points3D.get(id) is not None for id in image.point3D_ids
        ]
        visible_ids = image.point3D_ids[visible]

        if filter_output and len(visible_ids) < FILTER_MIN_VISIBLE:
            continue

        visible_xyzs = [points3D[id] for id in visible_ids]
        visible_xys = image.xys[visible]
        if resize:
            visible_xys *= scale_factor

        rr.set_time_sequence("frame", frame_idx)

        points = [point.xyz for point in visible_xyzs]
        point_colors = [point.rgb for point in visible_xyzs]
        point_errors = [point.error for point in visible_xyzs]

        # rr.log_scalar("plot/avg_reproj_err", np.mean(point_errors), color=[240, 45, 58])

        rr.log_points(
            "points", points, colors=point_colors, ext={"error": point_errors}
        )

        rr.log_rigid3(
            "camera",
            child_from_parent=camera_from_world,
            xyz="RDF",  # X=Right, Y=Down, Z=Forward
        )

        # Log camera intrinsics
        rr.log_pinhole(
            "camera/image",
            child_from_parent=intrinsics,
            width=camera.width,
            height=camera.height,
        )

        if resize:
            img = cv2.imread(str(image_file))
            img = cv2.resize(img, resize)
            jpeg_quality = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
            _, encimg = cv2.imencode(".jpg", img, jpeg_quality)
            rr.log_image_file("camera/image", img_bytes=encimg)
        else:
            img = cv2.imread(str(image_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rr.log_image(
                "camera/image", image=img
            )

        rr.log_points("camera/image/keypoints", visible_xys, colors=[34, 138, 167])

def main() -> None:
    parser = ArgumentParser(description="Visualize the output of COLMAP's sparse reconstruction on a video.")
    parser.add_argument("--unfiltered", action="store_true", help="If set, we don't filter away any noisy data.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        action="store",
        help="Which dataset to download",
    )
    parser.add_argument("--resize", action="store", help="Target resolution to resize images")
    rr.script_add_args(parser)
    args = parser.parse_args()

    if args.resize:
        args.resize = tuple(int(x) for x in args.resize.split("x"))

    rr.script_setup(args, "colmap")
    read_and_log_sparse_reconstruction(args.dataset_path, filter_output=not args.unfiltered, resize=args.resize)
    rr.script_teardown(args)


if __name__ == "__main__":
    main()