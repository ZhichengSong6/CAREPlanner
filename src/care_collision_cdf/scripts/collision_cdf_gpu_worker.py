#!/usr/bin/env python3
"""Persistent CUDA worker for CAREPlanner signed collision CDF.

The worker intentionally has no ROS dependency. It runs inside the CUDA-capable
viscdf environment and communicates with the ordinary ROS Python shadow node
through a Unix-domain stream socket.

Request:
  header: magic=b"CQ01", uint32 pair_count
  body:   float32 [B,10] rows = [px,py,pz,q0,...,q6]

Response:
  header: magic=b"CR01", uint32 pair_count,
          float64 h2d_ms, inference_ms, d2h_ms, worker_total_ms
  body:   float32 distance[B], float32 gradient[B,7]

This is a diagnostic transport, not the final C++ MPC integration.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from collision_cdf_model import CollisionCDF  # noqa: E402


REQ_MAGIC = b"CQ01"
RESP_MAGIC = b"CR01"
REQ_HEADER = struct.Struct("<4sI")
RESP_HEADER = struct.Struct("<4sI4d")


def recv_exact(conn: socket.socket, nbytes: int) -> bytes:
    chunks = []
    remaining = int(nbytes)
    while remaining > 0:
        chunk = conn.recv(remaining)
        if not chunk:
            raise EOFError("peer closed socket")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--checkpoint-key", default="latest")
    p.add_argument("--activation", choices=("gelu", "relu"), default="gelu")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--socket",
        default="/tmp/care_collision_cdf_gpu.sock",
    )
    p.add_argument("--max-pairs", type=int, default=8000)
    p.add_argument("--warmup-pairs", type=int, default=2048)
    p.add_argument("--socket-backlog", type=int, default=2)
    return p.parse_args()


class Worker:
    def __init__(self, args):
        if not str(args.device).startswith("cuda"):
            raise ValueError("C5.2g GPU worker requires a CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but torch.cuda.is_available() is False"
            )
        if args.max_pairs <= 0:
            raise ValueError("--max-pairs must be positive")

        self.args = args
        self.device = torch.device(args.device)
        self.socket_path = os.path.abspath(os.path.expanduser(args.socket))
        self.stop = False

        self.cdf = CollisionCDF(
            checkpoint_path=os.path.abspath(
                os.path.expanduser(args.checkpoint)
            ),
            device=args.device,
            checkpoint_key=args.checkpoint_key,
            input_dims=10,
            output_dims=1,
            nerf=True,
            activation=args.activation,
        )

        b = max(1, min(int(args.warmup_pairs), int(args.max_pairs)))
        p = torch.zeros((b, 3), dtype=torch.float32, device=self.device)
        q = torch.zeros((b, 7), dtype=torch.float32, device=self.device)
        with torch.enable_grad():
            self.cdf.pair_distance_and_gradient(p, q)
        torch.cuda.synchronize(self.device)

        print(
            "[C5.2g GPU worker] READY "
            f"gpu={torch.cuda.get_device_name(self.device)} "
            f"torch={torch.__version__} cuda={torch.version.cuda} "
            f"activation={self.cdf.architecture.get('activation')} "
            f"checkpoint={self.cdf.selected_checkpoint} "
            f"max_pairs={args.max_pairs} socket={self.socket_path}",
            flush=True,
        )

    def _handle_request(self, conn, pair_count, payload):
        if pair_count <= 0 or pair_count > self.args.max_pairs:
            raise ValueError(
                f"invalid pair_count={pair_count}, max={self.args.max_pairs}"
            )

        expected = pair_count * 10 * 4
        if len(payload) != expected:
            raise ValueError(
                f"payload bytes={len(payload)}, expected={expected}"
            )

        worker_t0 = time.perf_counter()

        rows = np.frombuffer(payload, dtype="<f4").reshape(pair_count, 10)
        points_cpu = torch.from_numpy(
            np.ascontiguousarray(rows[:, :3])
        )
        q_cpu = torch.from_numpy(
            np.ascontiguousarray(rows[:, 3:])
        )

        torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        points_gpu = points_cpu.to(self.device)
        q_gpu = q_cpu.to(self.device)
        torch.cuda.synchronize(self.device)
        t1 = time.perf_counter()

        with torch.enable_grad():
            distance_gpu, gradient_gpu = self.cdf.pair_distance_and_gradient(
                points_gpu, q_gpu
            )
        torch.cuda.synchronize(self.device)
        t2 = time.perf_counter()

        distance = (
            distance_gpu.cpu().numpy().astype("<f4", copy=False)
        )
        gradient = (
            gradient_gpu.cpu().numpy().astype("<f4", copy=False)
        )
        torch.cuda.synchronize(self.device)
        t3 = time.perf_counter()

        h2d_ms = (t1 - t0) * 1000.0
        inference_ms = (t2 - t1) * 1000.0
        d2h_ms = (t3 - t2) * 1000.0
        worker_total_ms = (t3 - worker_t0) * 1000.0

        header = RESP_HEADER.pack(
            RESP_MAGIC,
            pair_count,
            h2d_ms,
            inference_ms,
            d2h_ms,
            worker_total_ms,
        )
        conn.sendall(header)
        conn.sendall(distance.tobytes(order="C"))
        conn.sendall(gradient.tobytes(order="C"))

    def serve(self):
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(self.socket_path)
        server.listen(int(self.args.socket_backlog))
        server.settimeout(0.5)

        try:
            while not self.stop:
                try:
                    conn, _ = server.accept()
                except socket.timeout:
                    continue

                print("[C5.2g GPU worker] client connected", flush=True)
                with conn:
                    while not self.stop:
                        try:
                            header = recv_exact(conn, REQ_HEADER.size)
                        except EOFError:
                            break

                        magic, pair_count = REQ_HEADER.unpack(header)
                        if magic != REQ_MAGIC:
                            raise ValueError(
                                f"bad request magic: {magic!r}"
                            )

                        payload = recv_exact(
                            conn, int(pair_count) * 10 * 4
                        )
                        self._handle_request(
                            conn, int(pair_count), payload
                        )

                print("[C5.2g GPU worker] client disconnected", flush=True)
        finally:
            server.close()
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)

    def request_stop(self, *_args):
        self.stop = True


def main():
    args = parse_args()
    worker = Worker(args)
    signal.signal(signal.SIGINT, worker.request_stop)
    signal.signal(signal.SIGTERM, worker.request_stop)
    worker.serve()


if __name__ == "__main__":
    main()
