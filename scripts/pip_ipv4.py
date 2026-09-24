"""Run pip with IPv4 resolution confined to this installer process."""
import runpy
import socket

resolve = socket.getaddrinfo


def ipv4(host, port, family=0, type=0, proto=0, flags=0):
    return resolve(host, port, socket.AF_INET, type, proto, flags)


socket.getaddrinfo = ipv4
runpy.run_module("pip", run_name="__main__")
