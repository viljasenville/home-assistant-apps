#!/usr/bin/env python3
"""Decide which add-ons this push builds, and with what parameters.

The workflow does not list add-ons: every top-level directory holding a
config.yaml is one. This script turns that directory listing plus the event
that triggered the run into the build matrix, so adding an add-on means adding
a directory and nothing else.

Reads (all from the environment, set by the workflow):
    EVENT_NAME          github.event_name
    CHANGED_FILES_FILE  a file with one changed path per line
    INPUT_ADDON         workflow_dispatch: a slug, or "all"
    INPUT_PUBLISH       workflow_dispatch: "true" to push to GHCR
    GITHUB_REF_NAME     the tag name on a tag push
    GITHUB_REF_TYPE     "tag" or "branch"
    GITHUB_REPOSITORY   owner/repo, for the fallback image name and for
                        checking that a publish targets the owner's own packages

Writes to GITHUB_OUTPUT:
    matrix    {"include": [...]} for strategy.matrix
    tests     {"include": [...]} for the add-ons that ship .ci/test.sh
    publish   "true" / "false"
    any       "true" / "false"; the build job is skipped when false
    selected  human-readable summary for the step summary
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

# Archs are built natively: a LightGBM image under QEMU takes tens of minutes
# and times out. Each arch therefore needs a runner that executes it directly.
# 32-bit arm is absent on purpose — GitHub has no armv7 runner, and arm64
# runners are not guaranteed to execute 32-bit binaries.
RUNNERS = {
    "amd64": "ubuntu-24.04",
    "i386": "ubuntu-24.04",
    "aarch64": "ubuntu-24.04-arm",
}

# Touching any of these rebuilds everything: they can break any add-on.
SHARED_PREFIXES = (".github/", "repository.yaml")

TAG_RE = re.compile(r"^(?P<slug>.+)-v(?P<version>[0-9].*)$")


class Problem(SystemExit):
    """Aborts the run with a message GitHub renders as an annotation."""

    def __init__(self, message: str) -> None:
        super().__init__(f"::error::{message}")


@dataclass(frozen=True)
class Addon:
    slug: str
    dir: str
    version: str
    archs: list[str]
    image: str
    options: dict
    port: str
    bases: dict[str, str]
    has_tests: bool

    def base_for(self, arch: str) -> str:
        if self.bases:
            if arch not in self.bases:
                raise Problem(
                    f"{self.dir}/build.yaml has no build_from entry for {arch}, "
                    f"which {self.dir}/config.yaml lists under arch."
                )
            return self.bases[arch]
        # No build.yaml: Supervisor's own default base image.
        return f"ghcr.io/home-assistant/{arch}-base:latest"


def _read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise Problem(f"{path.relative_to(ROOT)} is not a YAML mapping.")
    return data


def _first_tcp_port(config: dict) -> str:
    """The port the CI health check probes, or "" for an add-on with no API.

    config.yaml's ports map "8099/tcp" to an optional host port. The container
    listens on the key regardless of whether a host port is published, and CI
    publishes it itself, so the key is what matters here.
    """
    for spec in (config.get("ports") or {}):
        text = str(spec)
        if text.endswith("/tcp"):
            return text[: -len("/tcp")]
    return ""


def load_addon(directory: Path) -> Addon:
    config = _read_yaml(directory / "config.yaml")
    rel = directory.name

    slug = str(config.get("slug") or "").strip()
    if not slug:
        raise Problem(f"{rel}/config.yaml has no slug.")
    if slug != rel:
        raise Problem(
            f"{rel}/config.yaml declares slug '{slug}'. The directory name and "
            "the slug must match, or a release tag cannot name the add-on."
        )

    version = str(config.get("version") or "").strip()
    if not version:
        raise Problem(f"{rel}/config.yaml has no version.")

    archs = [str(a) for a in (config.get("arch") or [])]
    if not archs:
        raise Problem(f"{rel}/config.yaml lists no arch.")
    unsupported = [a for a in archs if a not in RUNNERS]
    if unsupported:
        raise Problem(
            f"{rel}/config.yaml lists arch {', '.join(unsupported)}, which this "
            f"workflow cannot build natively. Supported: {', '.join(RUNNERS)}."
        )

    # GHCR rejects uppercase in a repository path, and an owner may well have
    # capitals in their account name.
    image = str(config.get("image") or "").strip()
    if not image:
        repo = os.environ.get("GITHUB_REPOSITORY", "local/repo")
        image = f"ghcr.io/{repo}/{{arch}}-addon-{slug}"
    if "{arch}" not in image:
        raise Problem(
            f"{rel}/config.yaml image '{image}' has no {{arch}} placeholder, so "
            "the architectures would overwrite each other's tags."
        )

    bases = {
        str(k): str(v)
        for k, v in (_read_yaml(directory / "build.yaml").get("build_from") or {}).items()
    } if (directory / "build.yaml").is_file() else {}

    return Addon(
        slug=slug,
        dir=rel,
        version=version,
        archs=archs,
        image=image.lower(),
        options=config.get("options") or {},
        port=_first_tcp_port(config),
        bases=bases,
        has_tests=(directory / ".ci" / "test.sh").is_file(),
    )


def check_publishable(addon: Addon, repository: str) -> None:
    """Refuse to publish to a GHCR path this repository's owner does not own.

    The common mistake is copying an add-on and tagging a release without
    changing config.yaml's image field: the push then fails deep in the build
    with a permission error, or — for someone who happens to have write access
    upstream — succeeds against the wrong package. Both are worth catching here.
    """
    if not addon.image.startswith("ghcr.io/"):
        print(
            f"::warning::{addon.slug} publishes to '{addon.image}', which is not "
            "on ghcr.io. The workflow only logs in to GHCR."
        )
        return
    owner = repository.split("/")[0].lower()
    if not owner:
        return
    path = addon.image[len("ghcr.io/"):]
    if path.split("/")[0] != owner:
        raise Problem(
            f"{addon.dir}/config.yaml publishes to '{addon.image}', which is not "
            f"under '{owner}'. Point the image field at this repository "
            f"(ghcr.io/{repository.lower()}/{{arch}}-addon-{addon.slug}) before "
            "tagging a release."
        )


def find_addons(root: Path = ROOT) -> dict[str, Addon]:
    addons = {}
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        if (entry / "config.yaml").is_file():
            addon = load_addon(entry)
            addons[addon.slug] = addon
    return addons


def changed_slugs(changed: list[str], addons: dict[str, Addon]) -> list[str]:
    """Which add-ons a set of changed paths affects."""
    if any(path.startswith(SHARED_PREFIXES) for path in changed):
        return sorted(addons)
    hit = {slug for slug in addons for path in changed if path.startswith(f"{slug}/")}
    return sorted(hit)


def read_changed(path: str | None) -> list[str]:
    if not path or not Path(path).is_file():
        return []
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def released_by_tag(tag: str, addons: dict[str, Addon]) -> list[str]:
    """The add-ons a release tag names, checking the version it claims."""
    match = TAG_RE.match(tag)
    if match and match["slug"] in addons:
        slug, version = match["slug"], match["version"]
        selected = [slug]
    elif tag.startswith("v") and len(addons) == 1:
        # A single-add-on repository may use a plain version tag.
        slug = next(iter(addons))
        version, selected = tag[1:], [slug]
    else:
        known = ", ".join(addons) or "none"
        raise Problem(
            f"Tag '{tag}' does not name an add-on in this repository. Use "
            f"<slug>-v<version>, e.g. '{next(iter(addons), 'myaddon')}-v1.0.0'. "
            f"Known add-ons: {known}."
        )

    for slug in selected:
        declared = addons[slug].version
        if declared != version:
            raise Problem(
                f"Tag '{tag}' releases version {version}, but "
                f"{slug}/config.yaml says {declared}. Bump the config, or "
                f"retag as {slug}-v{declared}."
            )
    return selected


def select(addons: dict[str, Addon], env: dict[str, str]) -> tuple[list[str], bool]:
    """The add-ons to build, and whether to publish them."""
    event = env.get("EVENT_NAME", "")

    if env.get("GITHUB_REF_TYPE") == "tag":
        return released_by_tag(env.get("GITHUB_REF_NAME", ""), addons), True

    if event == "workflow_dispatch":
        wanted = (env.get("INPUT_ADDON") or "all").strip()
        publish = (env.get("INPUT_PUBLISH") or "false").strip().lower() == "true"
        if wanted in ("", "all"):
            return sorted(addons), publish
        if wanted not in addons:
            raise Problem(
                f"No add-on '{wanted}' in this repository. "
                f"Known add-ons: {', '.join(addons) or 'none'}."
            )
        return [wanted], publish

    # push or pull_request: build what changed, publish nothing.
    return changed_slugs(read_changed(env.get("CHANGED_FILES_FILE")), addons), False


def build_matrix(addons: dict[str, Addon], selected: list[str]) -> dict:
    include = []
    for slug in selected:
        addon = addons[slug]
        for arch in addon.archs:
            include.append(
                {
                    "slug": slug,
                    "arch": arch,
                    "runner": RUNNERS[arch],
                    "dir": addon.dir,
                    "image": addon.image.replace("{arch}", arch),
                    "version": addon.version,
                    "base": addon.base_for(arch),
                    "options": json.dumps(addon.options, separators=(",", ":")),
                    "port": addon.port,
                }
            )
    return {"include": include}


def test_matrix(addons: dict[str, Addon], selected: list[str]) -> dict:
    return {
        "include": [
            {"slug": slug, "dir": addons[slug].dir}
            for slug in selected
            if addons[slug].has_tests
        ]
    }


def write_outputs(outputs: dict[str, str]) -> None:
    target = os.environ.get("GITHUB_OUTPUT")
    lines = [f"{key}={value}" for key, value in outputs.items()]
    if target:
        with open(target, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    for line in lines:
        print(line)


def main() -> int:
    addons = find_addons()
    if not addons:
        raise Problem("No add-on found: no top-level directory holds a config.yaml.")

    selected, publish = select(addons, dict(os.environ))
    if publish:
        repository = os.environ.get("GITHUB_REPOSITORY", "")
        for slug in selected:
            check_publishable(addons[slug], repository)

    matrix = build_matrix(addons, selected)
    tests = test_matrix(addons, selected)

    print(f"Add-ons in this repository: {', '.join(addons)}")
    if not selected:
        print("Nothing to build: no add-on's files changed.")
    for slug in selected:
        addon = addons[slug]
        print(f"Building {slug} {addon.version} for {', '.join(addon.archs)}")

    write_outputs(
        {
            "matrix": json.dumps(matrix, separators=(",", ":")),
            "tests": json.dumps(tests, separators=(",", ":")),
            "publish": "true" if publish else "false",
            "any": "true" if matrix["include"] else "false",
            "selected": ", ".join(f"{s} {addons[s].version}" for s in selected),
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
