"""Bound handshake admission in this node only; retain ROS topic/service handlers.

No system ROS files or global socket defaults are changed. Deadlines use wall
time so a paused simulation cannot leave an incomplete handshake in the queue.
"""
import socket
import struct
import threading
import time


class BoundedTCPROS:
    def __init__(self, server, decode, write, log, timeout_s=0.5,
                 max_pending=8, max_header_bytes=65536):
        if not 0 < timeout_s <= 5 or not 1 <= max_pending <= 64:
            raise ValueError('invalid TCPROS handshake admission limits')
        self.server, self.decode, self.write, self.log = server, decode, write, log
        self.timeout_s, self.max_header_bytes = timeout_s, max_header_bytes
        self.slots = threading.BoundedSemaphore(max_pending)
        self.lock = threading.Lock()
        self.pending = set()
        self.stopped = False

    def accept(self, sock, peer):
        # The accept thread never waits for a peer's handshake or a worker slot.
        started = time.monotonic()
        with self.lock:
            if self.stopped or not self.slots.acquire(blocking=False):
                sock.close()
                self.log('[OBS_TCPROS] rejected peer=%s reason=admission_capacity_or_shutdown', peer)
                return
            self.pending.add(sock)
        try:
            worker = threading.Thread(target=self._handle, args=(sock, peer, started), daemon=True)
            worker.start()
        except Exception:
            self._retire(sock)
            sock.close()
            raise

    def _retire(self, sock):
        with self.lock:
            self.pending.discard(sock)
        self.slots.release()

    def _header(self, sock, deadline, diagnostic):
        def read_exact(count):
            data = bytearray()
            while len(data) < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('handshake_deadline')
                sock.settimeout(remaining)
                part = sock.recv(count - len(data))
                if not part:
                    raise EOFError('incomplete_handshake')
                data.extend(part)
                diagnostic['received_bytes'] += len(part)
            return bytes(data)

        prefix = read_exact(4)
        size, = struct.unpack('<I', prefix)
        diagnostic['declared_bytes'] = size
        if not 0 < size <= self.max_header_bytes:
            raise ValueError('invalid_handshake_size')
        # Read exactly the header; never consume a following service request.
        return self.decode(prefix + read_exact(size))

    def _handle(self, sock, peer, started):
        diagnostic = dict(received_bytes=0, declared_bytes=None)
        header = {}
        try:
            header = self._header(sock, started + self.timeout_s, diagnostic)
            # Established ROS transports keep their ordinary blocking behavior.
            sock.settimeout(None)
            if 'topic' in header:
                error = self.server.topic_connection_handler(sock, peer, header)
            elif 'service' in header:
                error = self.server.service_connection_handler(sock, peer, header)
            else:
                error = 'no topic or service name detected'
            if error:
                sock.settimeout(max(.001, started + self.timeout_s - time.monotonic()))
                self.write(sock, {'error': str(error)})
                raise ValueError(str(error))
        except Exception as exc:
            self.log('[OBS_TCPROS] rejected peer=%s elapsed_wall_s=%.6f '
                     'received_bytes=%s declared_bytes=%s caller=%s topic=%s service=%s reason=%s',
                     peer, time.monotonic()-started, diagnostic['received_bytes'],
                     diagnostic['declared_bytes'], header.get('callerid', 'unknown'),
                     header.get('topic', 'none'), header.get('service', 'none'), str(exc))
            sock.close()
        finally:
            self._retire(sock)

    def shutdown(self):
        with self.lock:
            self.stopped = True
            pending = list(self.pending)
        for sock in pending:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()


def install_observation_tcp_admission():
    """Install before rospy.init_node creates its first publisher/service."""
    import rospy
    from rosgraph.network import decode_ros_handshake_header, write_ros_handshake_header
    from rospy.impl.tcpros_base import init_tcpros_server
    server = init_tcpros_server()
    if server.tcp_ros_server is not None:
        raise RuntimeError('observation TCPROS admission must precede server startup')
    guard = BoundedTCPROS(server, decode_ros_handshake_header,
                         write_ros_handshake_header, rospy.logwarn)
    server._tcp_server_callback = guard.accept
    rospy.on_shutdown(guard.shutdown)
    return guard
