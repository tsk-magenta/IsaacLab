import socket
import struct
import os

def send_image_and_get_completion(image_path, host='localhost', port=65432):
    """
    Connects to the paint completion server, sends an image, and receives the result.

    Args:
        image_path (str): The path to the image file to send.
        host (str): The server host address.
        port (int): The server port.

    Returns:
        str: The response message received from the server, or None if an error occurs.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None

    try:
        # Create a socket (AF_INET for IPv4, SOCK_STREAM for TCP)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            print(f"Connecting to server at {host}:{port}...")
            client_socket.connect((host, port))
            print("Connected to server.")

            # Read the image file in binary mode
            with open(image_path, 'rb') as f:
                image_data = f.read()

            image_size = len(image_data)
            print(f"Sending image data of size: {image_size} bytes")

            # Send the size of the image data first (4-byte unsigned integer)
            client_socket.sendall(struct.pack('!I', image_size))

            # Send the image data
            client_socket.sendall(image_data)
            print("Image data sent.")

            # Receive the size of the response message (4-byte unsigned integer)
            raw_response_size = client_socket.recv(4)
            if not raw_response_size:
                print("Server disconnected before sending response size.")
                return None

            response_size = struct.unpack('!I', raw_response_size)[0]
            print(f"Receiving response data of size: {response_size} bytes")

            # Receive the response message
            response_data = b''
            while len(response_data) < response_size:
                packet = client_socket.recv(response_size - len(response_data))
                if not packet:
                    print("Server disconnected during response data transfer.")
                    return None
                response_data += packet

            # Decode and return the response
            response_message = response_data.decode('utf-8')
            print(f"Received response: {response_message}")
            return response_message

    except ConnectionRefusedError:
        print(f"Error: Connection refused. Is the server running on {host}:{port}?")
        return None
    except FileNotFoundError:
         # This should be caught by the initial check, but good practice
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    # Example usage: Replace with the actual path to your test image
    # Make sure the server (detect_paintarea_server.py) is running before executing this client.
    test_image_path = "franka_spray.png" # <-- **CHANGE THIS PATH**

    # Check if the placeholder path is still there and warn the user
    if test_image_path == "path/to/your/test_image.png":
        print("\n--------------------------------------------------------------------")
        print("WARNING: Please update 'test_image_path' with the actual path to")
        print("         an image file you want to send to the server.")
        print("--------------------------------------------------------------------\n")
    else:
        completion_result = send_image_and_get_completion(test_image_path, host='127.0.0.1')

        if completion_result:
            print(f"\nFinal result from server: {completion_result}")
        else:
            print("\nFailed to get completion rate from server.")



