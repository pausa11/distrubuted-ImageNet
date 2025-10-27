import hivemind
dht = hivemind.DHT(start=True, host_maddrs=['/ip4/127.0.0.1/tcp/4000'], initial_peers=[])
print("/ip4/127.0.0.1/tcp/4000")
input("Bootstrap DHT corriendo. Enter para salir.\n")