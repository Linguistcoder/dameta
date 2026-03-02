import json
from pathlib import Path
MODEL_PREFIX = "openrouter/openai/gpt-4o-mini|"
def main():
    ckpt_path = Path("results") / "Danish Metaphor Benchmark v4_checkpoint.json"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    # Backup first
    backup_path = ckpt_path.with_suffix(".backup.json")
    backup_path.write_text(ckpt_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Backup written to: {backup_path}")
    data = json.loads(ckpt_path.read_text(encoding="utf-8"))
    items = data.get("processed_items", [])
    before = len(items)
    # Keep everything that is NOT gpt-4o-mini
    filtered = [k for k in items if not k.startswith(MODEL_PREFIX)]
    after = len(filtered)
    data["processed_items"] = filtered
    ckpt_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Total processed_items before: {before}")
    print(f"Total processed_items after : {after}")
    print(f"Removed {before - after} entries for model prefix {MODEL_PREFIX!r}")
if __name__ == "__main__":
    main()