import signal
import threading
from http.server import ThreadingHTTPServer


def run(hosts: list[str], port: int, handler, tls, allowed: set[str] | None) -> None:
    class Server(ThreadingHTTPServer):
        daemon_threads = True

        def verify_request(self, request, address) -> bool:
            return allowed is None or address[0] in allowed

    servers = []
    stopped = threading.Event()
    previous = {}
    try:
        for host in hosts:
            server = Server((host, port), handler)
            servers.append(server)
            server.socket = tls.wrap_socket(server.socket, server_side=True,
                                            do_handshake_on_connect=False)
            print(f"DPO labeler: https://{host}:{server.server_port}/", flush=True)
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, lambda *_: stopped.set())
        threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in servers]
        for thread in threads:
            thread.start()
        print("Trust the self-signed certificate / 請手動信任自簽憑證", flush=True)
        stopped.wait()
    finally:
        for server in servers:
            if any(t.is_alive() for t in locals().get("threads", [])):
                server.shutdown()
            server.server_close()
        for thread in locals().get("threads", []):
            thread.join()
        for sig, previous_handler in previous.items():
            signal.signal(sig, previous_handler)
