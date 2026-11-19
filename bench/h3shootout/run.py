"""H3 model shootout (Glia seed h3_model_shootout): the reference H3 clip through every installed H3 transformer.
Same prompt (6 s copy), start frame, seed, 720p 3:4, 8 steps, no LoRAs, encoder distilled 4B int8. Every model starts cold
(fresh WSL worker + dropped page cache). Records load / per-step / decode / total time, WSL RAM avg+peak (1 s), VRAM peak.
    .venv/Scripts/python.exe bench/h3shootout/run.py [model ...]      -> output/bench/h3shootout/<run>/status.json + <model>.mp4
"""
import json, shutil, subprocess, sys, threading, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from engine import jobs, wsl_runner

MODELS = {"flash_int8": "minimax_h3_ref2va_hyperflow",
          "gguf_q4_k_m": "minimax_h3_ref2va_hyperflow_gguf_q4_k_m",
          "gguf_q6_k": "minimax_h3_ref2va_hyperflow_gguf_q6_k",
          "10eros_q5_0": "minimax_h3_ref2va_10eros_turbo_q5_0",
          "10eros_q6_k": "minimax_h3_ref2va_10eros_turbo_q6_k"}
SECONDS, STEPS, RES, ASPECT = 6, 8, "720p", "3:4"


def prompt_6s() -> str:
    """The reference H3 prompt, re-timed for 6 s (shot 2 ends at 00:06). The master in reference_prompt.json stays 5 s."""
    p = json.loads((ROOT / "reference_prompt.json").read_text(encoding="utf-8"))["video"]["models"]["h3_minimax"]["prompt"]
    assert "00:05.000" in p, "reference prompt changed: re-check the 6 s re-timing"
    return p.replace("00:05.000", "00:06.000")


def settings(model_type: str, start: str, seed: int) -> dict:
    w, h = jobs.size(RES, ASPECT)
    s = {"model_type": model_type, "seed": seed, "resolution": f"{w}x{h}", "video_length": jobs.h3_frames(SECONDS),
         "force_fps": jobs.FPS, "multi_prompts_gen_type": "FG", "image_prompt_type": "S", "image_start": start,
         "prompt": jobs.h3_prompt(prompt_6s(), 1, None), "num_inference_steps": STEPS, "sample_solver": "euler",
         "guidance_scale": 1.0, "guidance_phases": 1, "config": "distilled_4b,lightvae_int8",
         "image_refs": [start], "video_prompt_type": "I"}
    if s["video_length"] < jobs.H3_WANGP_MIN:
        s["_frames_minimum"] = s["video_length"]
    return s


class RamSampler:
    """WSL used RAM (MemTotal - MemAvailable) once a second, from one long-lived wsl.exe."""
    def __init__(self):
        self.samples, self.cached, self.proc = [], [], None

    def start(self):
        self.samples, self.cached = [], []
        self.proc = subprocess.Popen(["wsl", "-e", "sh", "-c",
                                      "while :; do awk '/MemTotal/{t=$2}/MemAvailable/{a=$2}/^Cached/{c=$2}END{print t-a, t-a+c}' /proc/meminfo; sleep 1; done"],
                                     stdout=subprocess.PIPE, text=True)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            try:
                used, with_cache = (int(x) / 2**20 for x in line.split())  # kB -> GB
                self.samples.append(used); self.cached.append(with_cache)
            except ValueError:
                pass

    def stop(self):
        if self.proc:
            self.proc.kill(); self.proc = None

    def stats(self):
        s = self.samples
        return {"ram_avg_gb": round(sum(s) / len(s), 1) if s else None, "ram_peak_gb": round(max(s), 1) if s else None,
                "ram_now_gb": round(s[-1], 1) if s else None,
                # the weights sit in page cache (mmap), which MemAvailable counts as free: used + cache is the real footprint
                "ram_cache_avg_gb": round(sum(self.cached) / len(self.cached), 1) if self.cached else None,
                "ram_cache_peak_gb": round(max(self.cached), 1) if self.cached else None}


def main():
    ref = json.loads((ROOT / "reference_prompt.json").read_text(encoding="utf-8"))
    seed = ref["seed"]
    start = str((ROOT / ref["video"]["start_frame"]).resolve()).replace("\\", "/")
    names = [a for a in sys.argv[1:] if a in MODELS] or list(MODELS)
    into = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--into=")), None)  # add models to an earlier run
    out = Path(into) if into else ROOT / "output" / "bench" / "h3shootout" / time.strftime("%Y-%m-%d_%H%M"); out.mkdir(parents=True, exist_ok=True)
    status = json.loads((out / "status.json").read_text(encoding="utf-8")) if into else {"run": out.name, "settings": {"seconds": SECONDS, "steps": STEPS, "resolution": RES, "aspect": ASPECT,
                                             "seed": seed, "encoder": "distilled 4B int8", "loras": 0},
              "models": {}}
    status.pop("finished", None)
    status["models"].update({n: {"model_type": MODELS[n], "state": "queued"} for n in names})
    def save():
        (out / "status.json").write_text(json.dumps(status, indent=1), encoding="utf-8")
    save()
    ram = RamSampler()
    for n in names:
        m = status["models"][n]
        wsl_runner.release()  # cold: fresh worker per model ...
        subprocess.run(["wsl", "-u", "root", "-e", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"], check=False)  # ... and no page cache
        m.update(state="loading", started=time.time()); save()
        ram.start(); t0 = time.time(); marks = {}

        def on_event(ev):
            now = time.time() - t0
            if ev.get("phase") == "inference" and isinstance(ev.get("step"), int):
                marks.setdefault(f"step{ev['step']}", now)
                m.update(state="sampling", step=ev["step"], total=ev.get("total"))
            elif ev.get("phase") == "decoding":
                marks.setdefault("decode", now); m["state"] = "decoding"
            m.update(elapsed=round(now, 1), **ram.stats()); save()

        try:
            ev = wsl_runner.run({"kind": "render", "settings": settings(MODELS[n], start, seed)}, on_event, lambda: False, keep_warm=30)
            total = time.time() - t0
            dest = out / f"{n}.mp4"; shutil.copyfile(ev["file"], dest)
            d = Path(ev["file"]).parent
            if d.parent.name == "jobs":
                shutil.rmtree(d, ignore_errors=True); Path(str(d) + ".result.json").unlink(missing_ok=True)
            s0, dec = marks.get("step0"), marks.get("decode")
            steps = [marks[f"step{i + 1}"] - marks[f"step{i}"] for i in range(STEPS - 1) if f"step{i + 1}" in marks]
            if dec and f"step{STEPS - 1}" in marks:
                steps.append(dec - marks[f"step{STEPS - 1}"])  # last step ends when decoding starts
            m.update(state="done", video=dest.name, total_s=round(total, 1), worker_s=ev.get("seconds"),
                     load_s=round(s0, 1) if s0 else None,
                     step_first_s=round(steps[0], 1) if steps else None,
                     step_avg_s=round(sum(steps) / len(steps), 1) if steps else None,
                     step_avg_warm_s=round(sum(steps[1:]) / len(steps[1:]), 1) if len(steps) > 1 else None,
                     sampling_s=round(dec - s0, 1) if s0 and dec else None,
                     decode_s=round(total - dec, 1) if dec else None,
                     vram_peak_gb=ev.get("vram_peak_gb"), worker_ram_peak_gb=ev.get("ram_peak_gb"))
        except Exception as e:
            m.update(state="error", error=str(e))
        finally:
            ram.stop(); m.update(**{k: v for k, v in ram.stats().items() if k != "ram_now_gb"}); save()
        print(n, json.dumps(m), flush=True)
    wsl_runner.release()
    status["finished"] = time.time(); save()


if __name__ == "__main__":
    main()
