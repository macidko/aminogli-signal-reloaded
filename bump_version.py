import re

VERSION_FILE = "version.txt"

def bump_version(level="patch"):
    with open(VERSION_FILE, "r+") as f:
        version = f.read().strip()
        major, minor, patch = map(int, version.split("."))
        if level == "major":
            major += 1
            minor = 0
            patch = 0
        elif level == "minor":
            minor += 1
            patch = 0
        else:
            patch += 1
        new_version = f"{major}.{minor}.{patch}"
        f.seek(0)
        f.write(new_version + "\n")
        f.truncate()
    print(f"Version bumped to {new_version}")

if __name__ == "__main__":
    import sys
    level = sys.argv[1] if len(sys.argv) > 1 else "patch"
    bump_version(level)
