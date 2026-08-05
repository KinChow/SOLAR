# Third-party dependencies

`scripts/setup_locked_env.sh` creates `third_party/torchview` from:

- Repository: https://github.com/mert-kurttutan/torchview.git
- Commit: `edbe1fae97d94bb413c8fecace4f28b7ef9a59b0`
- Patch: `../patches/torchview-parameter-tensors.patch`

The checkout is ignored by Git and must not be edited manually. torchview keeps
its upstream license and is installed as an editable local dependency.
