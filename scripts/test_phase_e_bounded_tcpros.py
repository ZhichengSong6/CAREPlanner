#!/usr/bin/env python3
"""Bounded read semantics independent of ROS sockets and simulated time."""
import importlib.util
from pathlib import Path
import struct
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('bounded_tcpros',
    Path(__file__).resolve().parents[1]/'src/care_visibility_cdf/scripts/bounded_tcpros.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class Tests(unittest.TestCase):
    def run_header(self, data, chunk=9999, step=0.):
        now = [0.]
        class Peer:
            def settimeout(self, value): self.timeout = value
            def recv(self, n):
                count = min(n, chunk)
                result, self.data = self.data[:count], self.data[count:]
                now[0] += step
                return result
        peer = Peer(); peer.data = data
        guard = module.BoundedTCPROS(None, lambda b: b, None, None)
        diagnostic = dict(received_bytes=0)
        with patch.object(module.time, 'monotonic', lambda: now[0]):
            result = guard._header(peer, .5, diagnostic)
        return result, peer.data, diagnostic

    def test_full_header_preserves_following_service_payload(self):
        header = struct.pack('<I', 6)+b'abcdef'
        result, remaining, diagnostic = self.run_header(header+b'following_request')
        self.assertEqual(result, header)
        self.assertEqual(remaining, b'following_request')
        self.assertEqual(diagnostic['received_bytes'], 10)

    def test_fragmented_header_is_accepted(self):
        header = struct.pack('<I', 6)+b'abcdef'
        self.assertEqual(self.run_header(header, chunk=1)[0], header)

    def test_trickle_cannot_reset_absolute_deadline(self):
        with self.assertRaisesRegex(TimeoutError, 'handshake_deadline'):
            self.run_header(struct.pack('<I', 6)+b'abcdef', chunk=1, step=.2)

    def test_empty_and_oversized_declared_headers_rejected(self):
        for size in (0, 65537, 2**32-1):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, 'invalid_handshake_size'):
                self.run_header(struct.pack('<I', size))

    def test_eof_rejected(self):
        for data in (b'', b'abc', struct.pack('<I', 10)+b'abc'):
            with self.subTest(data=data), self.assertRaisesRegex(EOFError, 'incomplete_handshake'):
                self.run_header(data)


if __name__ == '__main__': unittest.main(verbosity=2)
