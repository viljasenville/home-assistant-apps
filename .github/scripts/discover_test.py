#!/usr/bin/env python3
"""Tests for discover.py. Run with: python3 .github/scripts/discover_test.py

Plain unittest, so CI needs nothing beyond the pyyaml discover.py already
requires. Each test builds a throwaway repository on disk rather than mocking
the filesystem, because the discovery it exercises is filesystem behaviour.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import discover  # noqa: E402

CONFIG = """\
---
name: Test add-on
slug: {slug}
version: "{version}"
arch:
{archs}
ports:
  {port}/tcp: null
options:
  log_level: info
  threads: 2
"""

BUILD = """\
---
build_from:
  amd64: ghcr.io/home-assistant/amd64-base-debian:bookworm
  aarch64: ghcr.io/home-assistant/aarch64-base-debian:bookworm
"""


def make_repo(tmp: Path, addons: dict) -> Path:
    """addons: slug -> dict(version, archs, port, build, tests, image)."""
    for slug, spec in addons.items():
        directory = tmp / slug
        directory.mkdir(parents=True)
        config = CONFIG.format(
            slug=spec.get("slug", slug),
            version=spec.get("version", "1.0.0"),
            archs="\n".join(f"  - {a}" for a in spec.get("archs", ["amd64"])),
            port=spec.get("port", 8099),
        )
        if "image" in spec:
            config += f"image: {spec['image']}\n"
        (directory / "config.yaml").write_text(config)
        if spec.get("build", True):
            (directory / "build.yaml").write_text(BUILD)
        if spec.get("tests"):
            (directory / ".ci").mkdir()
            (directory / ".ci" / "test.sh").write_text("#!/bin/sh\n")
    (tmp / "repository.yaml").write_text("---\nname: Test\n")
    (tmp / ".github").mkdir()
    return tmp


class RepoCase(unittest.TestCase):
    """Gives each test an empty repository to populate."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def addons(self, spec: dict) -> dict:
        return discover.find_addons(make_repo(self.tmp, spec))


class DiscoveryTest(RepoCase):
    def test_finds_every_directory_with_a_config(self):
        addons = self.addons({"alpha": {}, "beta": {}})
        self.assertEqual(sorted(addons), ["alpha", "beta"])

    def test_ignores_directories_without_a_config(self):
        make_repo(self.tmp, {"alpha": {}})
        (self.tmp / "docs").mkdir()
        (self.tmp / "docs" / "readme.md").write_text("hi")
        self.assertEqual(sorted(discover.find_addons(self.tmp)), ["alpha"])

    def test_slug_must_match_the_directory(self):
        with self.assertRaises(discover.Problem) as caught:
            self.addons({"alpha": {"slug": "something-else"}})
        self.assertIn("directory name", str(caught.exception))

    def test_unsupported_arch_is_rejected_with_a_readable_reason(self):
        with self.assertRaises(discover.Problem) as caught:
            self.addons({"alpha": {"archs": ["amd64", "armv7"]}})
        self.assertIn("armv7", str(caught.exception))

    def test_image_defaults_to_ghcr_and_is_lowercased(self):
        os.environ["GITHUB_REPOSITORY"] = "Owner/Repo"
        self.addCleanup(os.environ.pop, "GITHUB_REPOSITORY", None)
        addon = self.addons({"alpha": {}})["alpha"]
        self.assertEqual(addon.image, "ghcr.io/owner/repo/{arch}-addon-alpha")

    def test_image_needs_an_arch_placeholder(self):
        with self.assertRaises(discover.Problem):
            self.addons({"alpha": {"image": "ghcr.io/owner/repo/addon-alpha"}})

    def test_port_comes_from_the_tcp_key(self):
        self.assertEqual(self.addons({"alpha": {"port": 9123}})["alpha"].port, "9123")


class ChangedFilesTest(RepoCase):
    def test_only_the_changed_addon_is_selected(self):
        addons = self.addons({"alpha": {}, "beta": {}})
        self.assertEqual(
            discover.changed_slugs(["alpha/app/api.py", "README.md"], addons), ["alpha"]
        )

    def test_unrelated_change_selects_nothing(self):
        addons = self.addons({"alpha": {}})
        self.assertEqual(discover.changed_slugs(["README.md"], addons), [])

    def test_shared_files_rebuild_everything(self):
        addons = self.addons({"alpha": {}, "beta": {}})
        for path in (".github/workflows/build-addon.yml", "repository.yaml"):
            self.assertEqual(discover.changed_slugs([path], addons), ["alpha", "beta"])

    def test_a_prefix_is_not_a_directory_match(self):
        addons = self.addons({"alpha": {}})
        self.assertEqual(discover.changed_slugs(["alphabet/thing.py"], addons), [])


class TagTest(RepoCase):
    def test_slug_tag_selects_that_addon(self):
        addons = self.addons({"alpha": {"version": "2.1.0"}, "beta": {}})
        self.assertEqual(discover.released_by_tag("alpha-v2.1.0", addons), ["alpha"])

    def test_tag_version_must_match_the_config(self):
        addons = self.addons({"alpha": {"version": "2.1.0"}})
        with self.assertRaises(discover.Problem) as caught:
            discover.released_by_tag("alpha-v2.2.0", addons)
        self.assertIn("2.1.0", str(caught.exception))

    def test_slug_containing_a_dash_still_parses(self):
        addons = self.addons({"my-addon": {"version": "1.2.3"}})
        self.assertEqual(discover.released_by_tag("my-addon-v1.2.3", addons), ["my-addon"])

    def test_bare_version_tag_works_for_a_single_addon(self):
        addons = self.addons({"alpha": {"version": "1.0.0"}})
        self.assertEqual(discover.released_by_tag("v1.0.0", addons), ["alpha"])

    def test_bare_version_tag_is_ambiguous_with_two_addons(self):
        addons = self.addons({"alpha": {}, "beta": {}})
        with self.assertRaises(discover.Problem):
            discover.released_by_tag("v1.0.0", addons)

    def test_unknown_slug_fails_loudly(self):
        addons = self.addons({"alpha": {}})
        with self.assertRaises(discover.Problem):
            discover.released_by_tag("gamma-v1.0.0", addons)


class SelectionTest(RepoCase):
    def test_tag_push_publishes(self):
        addons = self.addons({"alpha": {"version": "1.0.0"}})
        selected, publish = discover.select(
            addons,
            {"EVENT_NAME": "push", "GITHUB_REF_TYPE": "tag", "GITHUB_REF_NAME": "alpha-v1.0.0"},
        )
        self.assertEqual((selected, publish), (["alpha"], True))

    def test_branch_push_never_publishes(self):
        addons = self.addons({"alpha": {}})
        changed = self.tmp / "changed.txt"
        changed.write_text("alpha/Dockerfile\n")
        selected, publish = discover.select(
            addons,
            {"EVENT_NAME": "push", "GITHUB_REF_TYPE": "branch",
             "CHANGED_FILES_FILE": str(changed)},
        )
        self.assertEqual((selected, publish), (["alpha"], False))

    def test_dispatch_can_publish_one_addon(self):
        addons = self.addons({"alpha": {}, "beta": {}})
        selected, publish = discover.select(
            addons,
            {"EVENT_NAME": "workflow_dispatch", "INPUT_ADDON": "beta",
             "INPUT_PUBLISH": "true"},
        )
        self.assertEqual((selected, publish), (["beta"], True))

    def test_dispatch_all(self):
        addons = self.addons({"alpha": {}, "beta": {}})
        selected, _ = discover.select(
            addons, {"EVENT_NAME": "workflow_dispatch", "INPUT_ADDON": "all"}
        )
        self.assertEqual(selected, ["alpha", "beta"])

    def test_dispatch_with_an_unknown_slug_fails(self):
        addons = self.addons({"alpha": {}})
        with self.assertRaises(discover.Problem):
            discover.select(
                addons, {"EVENT_NAME": "workflow_dispatch", "INPUT_ADDON": "nope"}
            )

    def test_missing_changed_file_list_selects_nothing(self):
        addons = self.addons({"alpha": {}})
        selected, _ = discover.select(
            addons, {"EVENT_NAME": "push", "CHANGED_FILES_FILE": "/nonexistent"}
        )
        self.assertEqual(selected, [])


class MatrixTest(RepoCase):
    def test_one_entry_per_arch_with_everything_the_build_needs(self):
        addons = self.addons(
            {"alpha": {"version": "3.0.0", "archs": ["amd64", "aarch64"]}}
        )
        include = discover.build_matrix(addons, ["alpha"])["include"]
        self.assertEqual([e["arch"] for e in include], ["amd64", "aarch64"])
        first = include[0]
        self.assertEqual(first["runner"], "ubuntu-24.04")
        self.assertEqual(include[1]["runner"], "ubuntu-24.04-arm")
        self.assertEqual(first["base"], "ghcr.io/home-assistant/amd64-base-debian:bookworm")
        self.assertEqual(first["version"], "3.0.0")
        self.assertEqual(first["port"], "8099")
        self.assertEqual(json.loads(first["options"]), {"log_level": "info", "threads": 2})
        self.assertIn("amd64", first["image"])

    def test_image_tags_differ_per_arch(self):
        addons = self.addons({"alpha": {"archs": ["amd64", "aarch64"]}})
        include = discover.build_matrix(addons, ["alpha"])["include"]
        self.assertNotEqual(include[0]["image"], include[1]["image"])

    def test_missing_build_from_entry_is_reported(self):
        # build.yaml covers amd64 and aarch64; config.yaml asks for i386.
        addons = self.addons({"alpha": {"archs": ["i386"]}})
        with self.assertRaises(discover.Problem) as caught:
            discover.build_matrix(addons, ["alpha"])
        self.assertIn("i386", str(caught.exception))

    def test_without_build_yaml_the_default_ha_base_is_used(self):
        addons = self.addons({"alpha": {"build": False}})
        base = discover.build_matrix(addons, ["alpha"])["include"][0]["base"]
        self.assertEqual(base, "ghcr.io/home-assistant/amd64-base:latest")

    def test_tests_matrix_holds_only_addons_shipping_a_test_script(self):
        addons = self.addons({"alpha": {"tests": True}, "beta": {}})
        include = discover.test_matrix(addons, ["alpha", "beta"])["include"]
        self.assertEqual(include, [{"slug": "alpha", "dir": "alpha"}])


class PublishTargetTest(RepoCase):
    def addon(self, image: str):
        return self.addons({"alpha": {"image": image}})["alpha"]

    def test_own_ghcr_path_is_accepted(self):
        addon = self.addon("ghcr.io/owner/repo/{arch}-addon-alpha")
        discover.check_publishable(addon, "Owner/Repo")  # no raise

    def test_someone_elses_ghcr_path_is_refused(self):
        addon = self.addon("ghcr.io/upstream/repo/{arch}-addon-alpha")
        with self.assertRaises(discover.Problem) as caught:
            discover.check_publishable(addon, "owner/repo")
        self.assertIn("upstream", str(caught.exception))

    def test_a_non_ghcr_registry_only_warns(self):
        addon = self.addon("docker.io/owner/{arch}-addon-alpha")
        discover.check_publishable(addon, "owner/repo")  # no raise

    def test_a_push_build_does_not_check_the_image_owner(self):
        # Nothing is published, so a fork building upstream's add-on is fine.
        make_repo(self.tmp, {"alpha": {"image": "ghcr.io/upstream/repo/{arch}-a"}})
        changed = self.tmp / "changed.txt"
        changed.write_text("alpha/Dockerfile\n")
        addons = discover.find_addons(self.tmp)
        _, publish = discover.select(
            addons,
            {"EVENT_NAME": "push", "GITHUB_REF_TYPE": "branch",
             "CHANGED_FILES_FILE": str(changed)},
        )
        self.assertFalse(publish)


class RealRepositoryTest(unittest.TestCase):
    """The add-ons actually in this repository must survive discovery."""

    def test_this_repository_is_discoverable(self):
        addons = discover.find_addons()
        self.assertTrue(addons, "no add-on found in this repository")
        repository = os.environ.get("GITHUB_REPOSITORY")
        for slug, addon in addons.items():
            include = discover.build_matrix(addons, [slug])["include"]
            self.assertEqual(len(include), len(addon.archs))
            for entry in include:
                self.assertTrue(entry["base"], f"{slug}/{entry['arch']} has no base image")
                json.loads(entry["options"])
            if repository:
                # Under Actions, confirm a release from here could publish.
                discover.check_publishable(addon, repository)


if __name__ == "__main__":
    unittest.main(verbosity=2)
