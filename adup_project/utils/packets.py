"""
Packet definitions for ADUP, RIP, and OSPF protocols.
Implements low-overhead packet formats with security authentication.
"""

import struct
import hashlib
import ipaddress
from typing import List, Optional, Union
from dataclasses import dataclass, field
from enum import IntEnum


class PacketType(IntEnum):
    """Packet type opcodes for ADUP protocol."""
    HELLO = 1
    UPDATE = 2
    QUERY = 3
    REPLY = 4


class RipCommand(IntEnum):
    """RIP command types."""
    REQUEST = 1
    RESPONSE = 2


class OspfType(IntEnum):
    """OSPF packet types."""
    HELLO = 1
    DATABASE_DESCRIPTION = 2
    LINK_STATE_REQUEST = 3
    LINK_STATE_UPDATE = 4
    LINK_STATE_ACK = 5


@dataclass
class CommonHeader:
    """
    Common header for ADUP packets (8 bytes total).
    
    Format:
    - Version (4 bits): Protocol version (1)
    - OpCode (4 bits): Message type (1:Hello, 2:Update, 3:Query, 4:Reply)
    - Flags (1 byte): Reserved for future use
    - Checksum (2 bytes): For error checking
    - Auth MAC (4 bytes): Message Authentication Code for security
    """
    version: int = 1
    opcode: PacketType = PacketType.HELLO
    flags: int = 0
    checksum: int = 0
    auth_mac: int = 0
    
    def pack(self) -> bytes:
        """Pack header into 8 bytes."""
        version_opcode = (self.version << 4) | (self.opcode & 0x0F)
        return struct.pack('!BBHI', version_opcode, self.flags, 
                          self.checksum, self.auth_mac)
    
    @classmethod
    def unpack(cls, data: bytes) -> 'CommonHeader':
        """Unpack 8 bytes into header."""
        version_opcode, flags, checksum, auth_mac = struct.unpack('!BBHI', data[:8])
        version = (version_opcode >> 4) & 0x0F
        opcode = PacketType(version_opcode & 0x0F)
        return cls(version, opcode, flags, checksum, auth_mac)


@dataclass
class AdupHelloPacket:
    """
    ADUP Hello packet for neighbor discovery and metric exchange.
    Total size: 14 bytes (Common Header 8 + Payload 6).
    
    Payload format:
    - Delay (2 bytes): Link delay in milliseconds
    - Jitter (2 bytes): Link jitter in microseconds  
    - Packet Loss (1 byte): Packet loss percentage (0-100)
    - Congestion (1 byte): Congestion level (0-100)
    """
    header: CommonHeader = field(default_factory=lambda: CommonHeader(opcode=PacketType.HELLO))
    delay: int = 0          # milliseconds
    jitter: int = 0         # microseconds
    packet_loss: int = 0    # percentage (0-100)
    congestion: int = 0     # level (0-100)
    
    def pack(self, pre_shared_key: str = "") -> bytes:
        """Pack packet into 14 bytes with authentication."""
        payload = struct.pack('!HHBB', self.delay, self.jitter, 
                             self.packet_loss, self.congestion)
        
        # Calculate MAC for authentication
        self.header.auth_mac = self._calculate_mac(payload, pre_shared_key)
        
        # Calculate checksum
        header_bytes = self.header.pack()
        self.header.checksum = self._calculate_checksum(header_bytes + payload)
        
        return self.header.pack() + payload
    
    @classmethod
    def unpack(cls, data: bytes) -> 'AdupHelloPacket':
        """Unpack 14 bytes into Hello packet."""
        header = CommonHeader.unpack(data[:8])
        delay, jitter, packet_loss, congestion = struct.unpack('!HHBB', data[8:14])
        return cls(header, delay, jitter, packet_loss, congestion)
    
    def _calculate_mac(self, payload: bytes, key: str) -> int:
        """Calculate Message Authentication Code."""
        mac_data = payload + key.encode('utf-8')
        hash_result = hashlib.md5(mac_data).digest()
        return struct.unpack('!I', hash_result[:4])[0]
    
    def _calculate_checksum(self, data: bytes) -> int:
        """Calculate simple checksum."""
        return sum(data) & 0xFFFF


@dataclass
class RouteEntry:
    """
    Route entry for ADUP Update and Reply packets (10 bytes each).
    
    Format:
    - Metric Flags (1 byte): Route status flags
    - Prefix Length (1 byte): Network prefix length
    - Destination (4 bytes): Destination network address
    - Total Cost (4 bytes): Path cost to destination
    """
    metric_flags: int = 0
    prefix_length: int = 24
    destination: int = 0     # IPv4 address as integer
    total_cost: int = 0
    
    def pack(self) -> bytes:
        """Pack route entry into 10 bytes."""
        return struct.pack('!BBII', self.metric_flags, self.prefix_length,
                          self.destination, self.total_cost)
    
    @classmethod
    def unpack(cls, data: bytes) -> 'RouteEntry':
        """Unpack 10 bytes into route entry."""
        metric_flags, prefix_length, destination, total_cost = struct.unpack('!BBII', data)
        return cls(metric_flags, prefix_length, destination, total_cost)


@dataclass
class AdupUpdatePacket:
    """
    ADUP Update packet for route advertisement.
    Total size: 8 + N * 10 bytes (Common Header + Variable Route Entries).
    """
    header: CommonHeader = field(default_factory=lambda: CommonHeader(opcode=PacketType.UPDATE))
    routes: List[RouteEntry] = field(default_factory=list)
    
    def pack(self, pre_shared_key: str = "") -> bytes:
        """Pack packet with authentication."""
        payload = b''.join(route.pack() for route in self.routes)
        
        # Calculate MAC
        self.header.auth_mac = self._calculate_mac(payload, pre_shared_key)
        
        # Calculate checksum
        header_bytes = self.header.pack()
        self.header.checksum = self._calculate_checksum(header_bytes + payload)
        
        return self.header.pack() + payload
    
    @classmethod
    def unpack(cls, data: bytes) -> 'AdupUpdatePacket':
        """Unpack data into Update packet."""
        header = CommonHeader.unpack(data[:8])
        routes = []
        
        # Parse route entries (10 bytes each)
        for i in range(8, len(data), 10):
            if i + 10 <= len(data):
                route = RouteEntry.unpack(data[i:i+10])
                routes.append(route)
        
        return cls(header, routes)
    
    def _calculate_mac(self, payload: bytes, key: str) -> int:
        """Calculate Message Authentication Code."""
        mac_data = payload + key.encode('utf-8')
        hash_result = hashlib.md5(mac_data).digest()
        return struct.unpack('!I', hash_result[:4])[0]
    
    def _calculate_checksum(self, data: bytes) -> int:
        """Calculate simple checksum."""
        return sum(data) & 0xFFFF


@dataclass
class AdupQueryPacket:
    """
    ADUP Query packet sent when a route goes "Active".
    Total size: 13 bytes (Common Header 8 + Payload 5).
    
    Payload format:
    - Prefix Length (1 byte): Network prefix length
    - Destination (4 bytes): Destination network being queried
    """
    header: CommonHeader = field(default_factory=lambda: CommonHeader(opcode=PacketType.QUERY))
    prefix_length: int = 24
    destination: int = 0     # IPv4 address as integer
    
    def pack(self, pre_shared_key: str = "") -> bytes:
        """Pack packet into 13 bytes with authentication."""
        payload = struct.pack('!BI', self.prefix_length, self.destination)
        
        # Calculate MAC
        self.header.auth_mac = self._calculate_mac(payload, pre_shared_key)
        
        # Calculate checksum
        header_bytes = self.header.pack()
        self.header.checksum = self._calculate_checksum(header_bytes + payload)
        
        return self.header.pack() + payload
    
    @classmethod
    def unpack(cls, data: bytes) -> 'AdupQueryPacket':
        """Unpack 13 bytes into Query packet."""
        header = CommonHeader.unpack(data[:8])
        prefix_length, destination = struct.unpack('!BI', data[8:13])
        return cls(header, prefix_length, destination)
    
    def _calculate_mac(self, payload: bytes, key: str) -> int:
        """Calculate Message Authentication Code."""
        mac_data = payload + key.encode('utf-8')
        hash_result = hashlib.md5(mac_data).digest()
        return struct.unpack('!I', hash_result[:4])[0]
    
    def _calculate_checksum(self, data: bytes) -> int:
        """Calculate simple checksum."""
        return sum(data) & 0xFFFF


@dataclass
class AdupReplyPacket:
    """
    ADUP Reply packet sent in response to a Query.
    Total size: 18 bytes (Common Header 8 + Single Route Entry 10).
    """
    header: CommonHeader = field(default_factory=lambda: CommonHeader(opcode=PacketType.REPLY))
    route: RouteEntry = field(default_factory=RouteEntry)
    
    def pack(self, pre_shared_key: str = "") -> bytes:
        """Pack packet into 18 bytes with authentication."""
        payload = self.route.pack()
        
        # Calculate MAC
        self.header.auth_mac = self._calculate_mac(payload, pre_shared_key)
        
        # Calculate checksum
        header_bytes = self.header.pack()
        self.header.checksum = self._calculate_checksum(header_bytes + payload)
        
        return self.header.pack() + payload
    
    @classmethod
    def unpack(cls, data: bytes) -> 'AdupReplyPacket':
        """Unpack 18 bytes into Reply packet."""
        header = CommonHeader.unpack(data[:8])
        route = RouteEntry.unpack(data[8:18])
        return cls(header, route)
    
    def _calculate_mac(self, payload: bytes, key: str) -> int:
        """Calculate Message Authentication Code."""
        mac_data = payload + key.encode('utf-8')
        hash_result = hashlib.md5(mac_data).digest()
        return struct.unpack('!I', hash_result[:4])[0]
    
    def _calculate_checksum(self, data: bytes) -> int:
        """Calculate simple checksum."""
        return sum(data) & 0xFFFF


# RIP Packet Definitions for Comparison

@dataclass
class RipEntry:
    """RIP routing table entry."""
    address_family: int = 2  # IPv4
    route_tag: int = 0
    ip_address: int = 0
    subnet_mask: int = 0
    next_hop: int = 0
    metric: int = 16  # Infinity
    
    def pack(self) -> bytes:
        """Pack RIP entry into 20 bytes."""
        return struct.pack('!HHIIII', self.address_family, self.route_tag,
                          self.ip_address, self.subnet_mask, 
                          self.next_hop, self.metric)
    
    @classmethod
    def unpack(cls, data: bytes) -> 'RipEntry':
        """Unpack 20 bytes into RIP entry."""
        fields = struct.unpack('!HHIIII', data)
        return cls(*fields)


@dataclass
class RipPacket:
    """RIP packet for comparison testing."""
    command: RipCommand = RipCommand.RESPONSE
    version: int = 2
    reserved: int = 0
    entries: List[RipEntry] = field(default_factory=list)
    
    def pack(self) -> bytes:
        """Pack RIP packet."""
        header = struct.pack('!BBH', self.command, self.version, self.reserved)
        payload = b''.join(entry.pack() for entry in self.entries)
        return header + payload
    
    @classmethod
    def unpack(cls, data: bytes) -> 'RipPacket':
        """Unpack data into RIP packet."""
        command, version, reserved = struct.unpack('!BBH', data[:4])
        entries = []
        
        for i in range(4, len(data), 20):
            if i + 20 <= len(data):
                entry = RipEntry.unpack(data[i:i+20])
                entries.append(entry)
        
        return cls(RipCommand(command), version, reserved, entries)


# OSPF Packet Definitions for Comparison

@dataclass
class OspfHeader:
    """OSPF packet header."""
    version: int = 2
    packet_type: OspfType = OspfType.HELLO
    packet_length: int = 0
    router_id: int = 0
    area_id: int = 0
    checksum: int = 0
    auth_type: int = 0
    authentication: int = 0
    
    def pack(self) -> bytes:
        """Pack OSPF header into 24 bytes."""
        return struct.pack('!BBHIIHHI', self.version, self.packet_type,
                          self.packet_length, self.router_id, self.area_id,
                          self.checksum, self.auth_type, self.authentication)
    
    @classmethod
    def unpack(cls, data: bytes) -> 'OspfHeader':
        """Unpack 24 bytes into OSPF header."""
        fields = struct.unpack('!BBHIIHHI', data[:24])
        return cls(fields[0], OspfType(fields[1]), *fields[2:])


@dataclass
class OspfHelloPacket:
    """OSPF Hello packet for comparison."""
    header: OspfHeader = field(default_factory=lambda: OspfHeader(packet_type=OspfType.HELLO))
    network_mask: int = 0
    hello_interval: int = 10
    options: int = 0
    router_priority: int = 1
    router_dead_interval: int = 40
    designated_router: int = 0
    backup_designated_router: int = 0
    neighbors: List[int] = field(default_factory=list)
    
    def pack(self) -> bytes:
        """Pack OSPF Hello packet."""
        hello_data = struct.pack('!IHBBHII', self.network_mask, self.hello_interval,
                                self.options, self.router_priority, 
                                self.router_dead_interval, self.designated_router,
                                self.backup_designated_router)
        
        neighbors_data = b''.join(struct.pack('!I', neighbor) for neighbor in self.neighbors)
        
        payload = hello_data + neighbors_data
        self.header.packet_length = 24 + len(payload)
        
        return self.header.pack() + payload
    
    @classmethod
    def unpack(cls, data: bytes) -> 'OspfHelloPacket':
        """Unpack data into OSPF Hello packet."""
        header = OspfHeader.unpack(data[:24])
        
        hello_fields = struct.unpack('!IHBBHII', data[24:44])
        neighbors = []
        
        for i in range(44, len(data), 4):
            if i + 4 <= len(data):
                neighbor = struct.unpack('!I', data[i:i+4])[0]
                neighbors.append(neighbor)
        
        return cls(header, *hello_fields, neighbors)


def authenticate_packet(packet_data: bytes, pre_shared_key: str, expected_mac: int) -> bool:
    """
    Verify packet authentication using pre-shared key.
    
    Args:
        packet_data: Raw packet data without header
        pre_shared_key: Pre-shared key for authentication
        expected_mac: Expected MAC value from packet header
    
    Returns:
        True if authentication succeeds, False otherwise
    """
    mac_data = packet_data + pre_shared_key.encode('utf-8')
    hash_result = hashlib.md5(mac_data).digest()
    calculated_mac = struct.unpack('!I', hash_result[:4])[0]
    return calculated_mac == expected_mac


def ip_to_int(ip_str: str) -> int:
    """Convert IP address string to integer."""
    return int(ipaddress.IPv4Address(ip_str))


def int_to_ip(ip_int: int) -> str:
    """Convert integer to IP address string."""
    return str(ipaddress.IPv4Address(ip_int)) 