"""Pre-read hook for disk weight sync on Modal Volumes.

Modal Volume writes from another container are not visible to an existing
mount until ``reload()`` — exactly the object-store-mount case SkyRL's
``weight_sync_disk_pre_read_hook`` exists for. This module is copied into the
inference image at ``/root/skyrl_modal_wsync_hooks.py`` (importable via
``PYTHONPATH=/root``) and referenced in the init info as
``"skyrl_modal_wsync_hooks:reload_shared_volume"``. The volume name is passed
via the ``SKYRL_WSYNC_VOLUME`` env var baked into the image.
"""

import os


def reload_shared_volume() -> None:
    import modal

    volume_name = os.environ["SKYRL_WSYNC_VOLUME"]
    modal.Volume.from_name(volume_name, version=2).reload()
