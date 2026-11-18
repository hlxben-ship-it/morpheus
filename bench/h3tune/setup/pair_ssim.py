import sys; sys.path.insert(0, "/mnt/c/dev/morpheus/bench/h3tune"); import quality
O = "/mnt/c/dev/morpheus/output/bench/h3tune"
for a, b in [p.split(":") for p in sys.argv[1:]]: print(a, "vs", b, quality.check_png(f"{O}/{a}/{a}.png", f"{O}/{b}/{b}.png"))
