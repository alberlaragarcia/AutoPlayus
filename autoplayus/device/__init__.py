from autoplayus.device.base import Device


def create_device() -> Device:
    """Dispositivo por defecto: BlueStacks (captura por ADB, toques como clics de Windows)."""
    from autoplayus.device.bluestacks import BlueStacksDevice

    return BlueStacksDevice()
