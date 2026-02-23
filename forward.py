import socket
import threading

LISTEN_PORT = 8080
TARGET_HOST = "127.0.0.1"
TARGET_PORT = 8501

def handle(client):
    target = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    target.connect((TARGET_HOST, TARGET_PORT))

    threading.Thread(target=pipe, args=(client, target)).start()
    threading.Thread(target=pipe, args=(target, client)).start()

def pipe(src, dst):
    while True:
        data = src.recv(4096)
        if not data:
            break
        dst.sendall(data)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("0.0.0.0", LISTEN_PORT))
server.listen()

while True:
    client, _ = server.accept()
    threading.Thread(target=handle, args=(client,)).start()