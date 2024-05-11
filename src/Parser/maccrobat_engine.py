import asyncio
import struct
import json
import argparse

async def async_send_text_to_server(host, port, text):
    reader, writer = await asyncio.open_connection(host, port)

    message = text.encode('utf-8')
    message_length = struct.pack('>H', len(message))
    writer.write(message_length + message)
    await writer.drain()

    response_length_data = await reader.readexactly(2)
    if len(response_length_data) < 2:
        print("Error: Server sent an incomplete response.")
        writer.close()
        await writer.wait_closed()
        return {}

    response_length = struct.unpack('>H', response_length_data)[0]
    response_data = await reader.readexactly(response_length)

    writer.close()
    await writer.wait_closed()
    return json.loads(response_data.decode('utf-8'))

async def main():
    parser = argparse.ArgumentParser(description="Client for server communication")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument("--text", type=str, help="Text to analyze")
    args = parser.parse_args()

    entities = await async_send_text_to_server(args.host, args.port, args.text)
    print("Received entities from server:", json.dumps(entities, indent=4))

if __name__ == "__main__":
    asyncio.run(main())
