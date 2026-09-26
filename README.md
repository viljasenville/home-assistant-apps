# Home Assistant Apps

A Home Assistant add-on repository.

## Add-ons

| Add-on | Slug | What it does |
| --- | --- | --- |
| [Consumption Forecast Model Provider](cfmp/) | `cfmp` | Trains and serves LightGBM consumption forecast models over HTTP |

## Installation

In Home Assistant: **Settings → Add-ons → Add-on store → ⋮ → Repositories**,
then add:

```
https://github.com/viljasenville/home-assistant-apps
```

## Releasing

Tag the add-on's slug and version; the tag must match the add-on's
`config.yaml` or the run fails before building anything:

```
git tag cfmp-v2026.9.0
git push origin cfmp-v2026.9.0
```

A push or pull request builds only the add-ons whose files changed, without
publishing. See [.github/workflows/build-addon.yml](.github/workflows/build-addon.yml).

## License

[Apache-2.0](LICENSE)
