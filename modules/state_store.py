import json, os, time

class StateStore:
    def __init__(self, config):
        self.status_path = config["runtime"]["status_file"]
        self.journal_path = config["runtime"]["journal_file"]
        os.makedirs(os.path.dirname(self.status_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(self.journal_path) or ".", exist_ok=True)

    def write_status(self, payload):
        payload = dict(payload)
        payload["updated_at"] = time.time()
        tmp = self.status_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        os.replace(tmp, self.status_path)

    def journal(self, payload):
        with open(self.journal_path, "a") as f:
            f.write(json.dumps(payload, default=str) + "\n")
