import logging
import socket
import threading
import json
from .net_package_handler import PackageHandle, TrackingDecoder, NetPacket

logger = logging.getLogger(__name__)

class MetaQuest3Server:
    """
    用于meta quest3头显连接并接收头显数据的tcp服务器
    """
    def __init__(self, host='localhost', port=63901, teleoperator=None):
        self.host = host
        self.port = port
        self.server_socket = None
        self.running = False
        self.clients = []
        self.heartbeat_count = 0
        self.controller_count = 0
        self.client_buffers = {}
        self.teleoperator = teleoperator  # Reference to the teleoperator

    def start(self):
        """Start the TCP server"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(socket.SOMAXCONN)
            self.running = True

            logger.info("Server listening on %s:%d" % (self.host, self.port))
            logger.info("Waiting for connections...")

            while self.running:
                try:
                    client_socket, client_address = self.server_socket.accept()
                    logger.info(f"New connection from {client_address}")

                    # Create a new thread for each client
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, client_address)
                    )
                    client_thread.daemon = True
                    client_thread.start()

                    self.clients.append((client_socket, client_address, client_thread))

                except socket.error as e:
                    if self.running:
                        print(f"Socket error: {e}")
                    break

        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            self.stop()

    def handle_client(self, client_socket, client_address):
        """Handle individual client connection"""
        try:
            # Initialize bytearray buffer for this client
            self.client_buffers[client_address] = {'buffer': bytearray(), 'read_index': 0}

            while self.running:
                # Receive data from client
                data = client_socket.recv(4096)
                if not data:
                    break

                # Add data to client's buffer
                self.client_buffers[client_address]['buffer'].extend(data)

                # Process complete packets from buffer
                self.process_client_buffer(client_address, client_socket)

        except Exception as e:
            logger.error(f"Client {client_address} error: {e}")
        finally:
            # Clean up client buffer
            if client_address in self.client_buffers:
                del self.client_buffers[client_address]
            client_socket.close()
            logger.info(f"Client {client_address} disconnected")

    def process_client_buffer(self, client_address, client_socket):
        """Process complete packets from client's bytearray buffer"""
        buffer_info = self.client_buffers[client_address]
        buffer = buffer_info['buffer']
        read_index = buffer_info['read_index']

        while True:
            # Try to unpack a complete packet
            packet, new_read_index = PackageHandle.unpack_from_buffer(buffer, read_index)
            if packet is None:
                # No complete packet available, wait for more data
                break

            # Update read index
            buffer_info['read_index'] = new_read_index
            read_index = new_read_index

            # Handle the packet
            self.handle_packet(packet, client_address, client_socket)

            # Clean up processed data from buffer
            if read_index > 0:
                # Remove processed data from beginning of buffer
                remaining_data = buffer[read_index:]
                buffer.clear()
                buffer.extend(remaining_data)
                buffer_info['read_index'] = 0
                read_index = 0

    def handle_packet(self, packet: NetPacket, client_address, client_socket):
        """Handle a complete packet"""
        try:
            # Handle different packet types
            if PackageHandle.is_heartbeat_packet(packet):
                self.handle_heartbeat_packet(packet, client_address)
            elif PackageHandle.is_controller_packet(packet):
                self.handle_controller_packet(packet, client_address)
            else:
                print(f"Unknown packet type: 0x{packet.cmd:02X}")

        except Exception as e:
            print(f"Packet handling error: {e}")

    def handle_heartbeat_packet(self, packet: NetPacket, client_address):
        """Handle heartbeat packet (0x23)"""
        self.heartbeat_count += 1

        if packet.body:
            try:
                heartbeat_data = json.loads(packet.body.decode('utf-8'))
            except Exception as e:
                print(f"Heartbeat data decode error: {e}")

    def handle_controller_packet(self, packet: NetPacket, client_address, print_tracking_data: bool = False):
        """Handle controller function packet (0x6D)"""
        self.controller_count += 1

        if packet.body:
            try:
                # Try to decode as JSON
                json_str = packet.body.decode('utf-8')
                json_data = json.loads(json_str)

                # Check if it's a tracking message
                if 'functionName' in json_data and json_data['functionName'] == 'Tracking':
                    decoded = TrackingDecoder.decode_full_tracking_data(json_str)
                    if print_tracking_data:
                        self.print_tracking_data(decoded)
                    # Update teleoperator with tracking data
                    if self.teleoperator:
                        self.teleoperator.update_tracking_data(decoded)

            except Exception as e:
                print(f"Controller data decode error: {e}")

    def print_tracking_data(self, decoded_data):
        if 'data' in decoded_data:
            data = decoded_data['data']
            left_hand_data = data['Controller']['left']
            pose = left_hand_data['parsed_pose']
            print(f"    Position: {pose['position']} Rotation : {pose['rotation']}")

    def stop(self):
        """Stop the TCP server"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()

    @property
    def is_connected(self):
        return self.server_socket is not None
