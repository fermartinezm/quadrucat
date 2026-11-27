"""Record a Gazebo camera topic to an MP4 (run next to a `gazebo.launch.py video_dir:=...` launch).

Gazebo only renders a camera sensor while something subscribes to its image topic, so the
sensor's own <save> writes nothing in a headless server; this script subscribes and writes the
frames itself (raw RGB piped to ffmpeg).

  python scripts/record_gazebo_video.py out.mp4 --seconds 30
"""
from __future__ import annotations

import argparse
import subprocess
import time

TOPIC = "/world/flat/model/go1/link/trunk/sensor/video_cam/image"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--topic", default=TOPIC)
    ap.add_argument("--seconds", type=float, default=30.0, help="wall-clock recording time")
    ap.add_argument("--fps", type=int, default=30, help="camera update_rate (sim time)")
    args = ap.parse_args()

    from gz.msgs10.image_pb2 import Image
    from gz.transport13 import Node

    state = {"proc": None, "n": 0}

    def on_image(msg: Image) -> None:
        if state["proc"] is None:
            state["proc"] = subprocess.Popen(
                ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{msg.width}x{msg.height}", "-r", str(args.fps), "-i", "-",
                 "-pix_fmt", "yuv420p", "-vcodec", "libx264", args.out], stdin=subprocess.PIPE)
        state["proc"].stdin.write(msg.data)
        state["n"] += 1

    node = Node()
    if not node.subscribe(Image, args.topic, on_image):
        raise SystemExit(f"cannot subscribe to {args.topic}")
    t_end = time.time() + args.seconds
    while time.time() < t_end:
        time.sleep(0.1)
    node.unsubscribe(args.topic)
    time.sleep(0.5)
    if state["proc"] is None:
        raise SystemExit("no frames received (is the launch running with video_dir:=...?)")
    state["proc"].stdin.close()
    state["proc"].wait()
    print(f"[video] {state['n']} frames -> {args.out}")


if __name__ == "__main__":
    main()
