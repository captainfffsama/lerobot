"""
Teleaent Packet Decoder V2
Based on C# Network protocol definitions
"""

import struct
import json
import time
from typing import Optional, Dict, Any


class NetPacket:
    """Network packet data structure"""

    def __init__(self, cmd: int, body: Optional[bytes], timestamp: int):
        self.cmd = cmd
        self.body = body
        self.timestamp = timestamp


class PackageHandle:
    """Packet handler class based on C# Network protocol"""

    # Packet constants from C# NetCMD.cs
    DEFAULT_PACKAGE_SIZE = 15
    SEND_PACKET_HEAD = 0x3F
    SEND_PACKET_END = 0xA5

    # Command constants from C# NetCMD.cs
    PACKET_CCMD_TO_CONTROLLER_FUNCTION = 0x6D  # General message returned to the control end
    PACKET_CCMD_CLIENT_HEARTBEAT = 0x23  # Client heartbeat

    @staticmethod
    def unpack(data: bytes) -> NetPacket:
        """
        Unpack network packet data according to C# ByteBuffer.Unpack method:
        head(1) + cmd(1) + params_len(4) + params(n) + timestamp(8) + end(1)

        Args:
            data: Raw packet bytes to decode

        Returns:
            NetPacket object containing decoded data
        """
        if not data or len(data) < PackageHandle.DEFAULT_PACKAGE_SIZE:
            raise ValueError("Packet too short or empty")

        # Check header - using SEND_PACKET_HEAD (0x3F) as per C# code
        if data[0] != PackageHandle.SEND_PACKET_HEAD:
            raise ValueError(
                f"Invalid packet header: expected 0x{PackageHandle.SEND_PACKET_HEAD:02X}, got 0x{data[0]:02X}")

        cmd = data[1]

        # Read parameter length (4 bytes, little-endian) - same as C# BitConverter.ToInt32
        length = struct.unpack('<I', data[2:6])[0]

        # Check if we have enough data for complete packet
        if len(data) < 15 + length:
            raise ValueError(f"Insufficient packet data: need {15 + length} bytes, got {len(data)}")

        # Check end marker
        end_byte = data[2 + 4 + length + 8]
        if end_byte != PackageHandle.SEND_PACKET_END:
            raise ValueError(
                f"Invalid packet end: expected 0x{PackageHandle.SEND_PACKET_END:02X}, got 0x{end_byte:02X}")

        # Extract parameter data (same as C# Buffer.BlockCopy)
        body = data[2 + 4:2 + 4 + length] if length > 0 else None

        # Extract timestamp (8 bytes, little-endian) - same as C# BitConverter.ToInt64
        timestamp = struct.unpack('<Q', data[2 + 4 + length:2 + 4 + length + 8])[0]

        return NetPacket(cmd=cmd, body=body, timestamp=timestamp)

    @staticmethod
    def unpack_from_buffer(buffer: bytearray, read_index: int) -> tuple[Optional[NetPacket], int]:
        """
        Unpack packet from bytearray buffer (similar to C# ByteBuffer.Unpack method)

        Args:
            buffer: bytearray containing packet data
            read_index: current read position in buffer

        Returns:
            tuple of (NetPacket object or None, new_read_index)
        """
        if len(buffer) - read_index < 15:
            return None, read_index

        # Check header
        head = buffer[read_index]
        if head != PackageHandle.SEND_PACKET_HEAD:
            print(f"Receive data head error! {head}")
            return None, len(buffer)  # Skip all remaining data

        cmd = buffer[read_index + 1]
        length = struct.unpack('<I', buffer[read_index + 2:read_index + 6])[0]

        # Check if we have enough data
        if len(buffer) - read_index < 15 + length:
            if length > len(buffer):
                print(f"Receive data length error! {length}")
                return None, len(buffer)  # Skip all remaining data
            return None, read_index  # Wait for more data

        # Check end marker
        end = buffer[read_index + 2 + 4 + length + 8]
        if end != PackageHandle.SEND_PACKET_END:
            print(f"Receive data end error! {end}")
            return None, len(buffer)  # Skip all remaining data

        # Extract parameter data
        data = bytes(buffer[read_index + 2 + 4:read_index + 2 + 4 + length])

        # Extract timestamp (8 bytes, little-endian)
        timestamp = struct.unpack('<Q', buffer[read_index + 2 + 4 + length:read_index + 2 + 4 + length + 8])[0]

        # Update read index
        new_read_index = read_index + 2 + 4 + length + 8 + 1

        return NetPacket(cmd=cmd, body=data, timestamp=timestamp), new_read_index

    @staticmethod
    def pack(cmd: int, message: bytes) -> bytes:
        """
        Pack data into network packet format

        Args:
            cmd: Command byte
            message: Message data to encode

        Returns:
            Packed packet bytes
        """
        data = bytearray(15 + len(message))

        # Header
        data[0] = PackageHandle.SEND_PACKET_HEAD

        # Command
        data[1] = cmd

        # Parameter length (4 bytes, little-endian)
        struct.pack_into('<I', data, 2, len(message))

        # Parameter data
        data[6:6 + len(message)] = message

        # Timestamp (8 bytes, little-endian)
        timestamp = int(time.time() * 1000)  # Milliseconds
        struct.pack_into('<Q', data, 6 + len(message), timestamp)

        # End marker
        data[-1] = PackageHandle.SEND_PACKET_END

        return bytes(data)

    @staticmethod
    def is_heartbeat_packet(packet: NetPacket) -> bool:
        """Check if packet is a heartbeat packet"""
        return packet.cmd == PackageHandle.PACKET_CCMD_CLIENT_HEARTBEAT

    @staticmethod
    def is_controller_packet(packet: NetPacket) -> bool:
        """Check if packet is a controller function packet"""
        return packet.cmd == PackageHandle.PACKET_CCMD_TO_CONTROLLER_FUNCTION


class TrackingDecoder:
    """Tracking data decoder for controller messages"""

    @staticmethod
    def decode_tracking_json(json_str: str) -> Dict[str, Any]:
        """Decode tracking JSON data"""
        return json.loads(json_str)

    @staticmethod
    def parse_controller_data(controller_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse controller data from tracking information"""
        result = {}

        for hand in ['left', 'right']:
            if hand in controller_data:
                hand_data = controller_data[hand]
                result[hand] = {
                    'axisX': hand_data.get('axisX', 0.0),
                    'axisY': hand_data.get('axisY', 0.0),
                    'axisClick': hand_data.get('axisClick', False),
                    'grip': hand_data.get('grip', 0.0),
                    'trigger': hand_data.get('trigger', 0.0),
                    'primaryButton': hand_data.get('primaryButton', False),
                    'secondaryButton': hand_data.get('secondaryButton', False),
                    'menuButton': hand_data.get('menuButton', False),
                    'pose': hand_data.get('pose', '0,0,0,0,0,0,-1')
                }

        return result

    @staticmethod
    def parse_pose_string(pose_str: str) -> Dict[str, Any]:
        """Parse pose string into position and rotation components"""
        parts = pose_str.split(',')
        if len(parts) != 7:
            raise ValueError("Invalid pose string format")

        return {
            'position': {
                'x': float(parts[0]),
                'y': float(parts[1]),
                'z': float(parts[2])
            },
            'rotation': {
                'x': float(parts[3]),
                'y': float(parts[4]),
                'z': float(parts[5]),
                'w': float(parts[6])
            }
        }

    @staticmethod
    def decode_full_tracking_data(json_str: str) -> Dict[str, Any]:
        """Decode complete tracking data including nested JSON"""
        # Parse outer JSON
        outer_data = json.loads(json_str)

        # Extract and parse inner JSON
        if 'value' in outer_data:
            inner_data = json.loads(outer_data['value'])

            # Parse controller data
            if 'Controller' in inner_data:
                inner_data['Controller'] = TrackingDecoder.parse_controller_data(inner_data['Controller'])

            # Parse pose strings
            if 'Controller' in inner_data:
                for hand in ['left', 'right']:
                    if hand in inner_data['Controller'] and 'pose' in inner_data['Controller'][hand]:
                        pose_str = inner_data['Controller'][hand]['pose']
                        inner_data['Controller'][hand]['parsed_pose'] = TrackingDecoder.parse_pose_string(pose_str)

            return {
                'functionName': outer_data.get('functionName'),
                'data': inner_data
            }
        else:
            return outer_data