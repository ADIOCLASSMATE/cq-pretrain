"""Select the existing IPv4 interface used to reach the rendezvous host."""
import fcntl
import socket
import struct
from pathlib import Path


def rendezvous_interface(master):
    addresses = {}
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
        for _, interface in socket.if_nameindex():
            try:
                info = fcntl.ioctl(route.fileno(), 0x8915, struct.pack('256s', interface.encode()[:15]))
                addresses[interface] = socket.inet_ntoa(info[20:24])
            except OSError:
                continue
        try:
            route.connect((socket.gethostbyname(master), 9))
            source = route.getsockname()[0]
        except socket.gaierror:
            # Job DNS can be installed after container startup. torchrun retries
            # rendezvous resolution; choose the kernel default route meanwhile.
            defaults = []
            for row in Path('/proc/net/route').read_text().splitlines()[1:]:
                fields = row.split()
                if fields[1] == '00000000' and fields[0] in addresses:
                    defaults.append((int(fields[6]), fields[0]))
            if defaults:
                return min(defaults)[1], addresses
            raise
    matches = [name for name, address in addresses.items() if address == source]
    if not matches:
        raise RuntimeError(f'Cannot map rendezvous source {source} to a local interface: {addresses}')
    return matches[0], addresses
