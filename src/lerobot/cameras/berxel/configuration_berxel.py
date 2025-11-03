from dataclasses import dataclass
from ..configs import CameraConfig, ColorMode, Cv2Rotation


@CameraConfig.register_subclass("berxel")
@dataclass
class BerxelCameraConfig(CameraConfig):
    """Configuration class for Berxel Hawk / P150E network cameras.

    This configuration defines parameters for Berxel network cameras (e.g., Hawk P150E),
    supporting RGB and optional depth streams over Ethernet.

    Example usage:
    ```python
    # Basic RGB configuration
    BerxelCameraConfig(ip="192.168.2.11", fps=30, width=1280, height=720)

    # With depth enabled
    BerxelCameraConfig(ip="192.168.2.11", fps=30, width=1280, height=720, use_depth=True)

    # With rotation and BGR color output
    BerxelCameraConfig(
        ip="192.168.2.11",
        fps=30,
        width=1280,
        height=720,
        color_mode=ColorMode.BGR,
        rotation=Cv2Rotation.ROTATE_90,
    )
    ```

    Attributes:
        ip: IPv4 address of the Berxel network camera.
        fps: Desired frames per second (FPS).
        width: Frame width in pixels.
        height: Frame height in pixels.
        color_mode: Output color format (RGB or BGR). Defaults to RGB.
        use_depth: Whether to enable depth stream. Defaults to True.
        rotation: Image rotation setting. Defaults to no rotation.
        warmup_s: Warm-up time before frame reading (in seconds). Defaults to 1.
    """

    ip: str = "192.168.2.11"
    fps: int = 30               # ✅ 新增
    width: int = 1280           # ✅ 新增
    height: int = 720           # ✅ 新增
    color_mode: ColorMode = ColorMode.RGB
    use_depth: bool = True
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1

    def __post_init__(self):
        """Validates color mode, rotation, and dimension consistency."""
        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"`color_mode` must be {ColorMode.RGB.value} or {ColorMode.BGR.value}, "
                f"but got {self.color_mode}."
            )

        if self.rotation not in (
            Cv2Rotation.NO_ROTATION,
            Cv2Rotation.ROTATE_90,
            Cv2Rotation.ROTATE_180,
            Cv2Rotation.ROTATE_270,
        ):
            raise ValueError(
                f"`rotation` must be one of "
                f"{(Cv2Rotation.NO_ROTATION, Cv2Rotation.ROTATE_90, Cv2Rotation.ROTATE_180, Cv2Rotation.ROTATE_270)}, "
                f"but got {self.rotation}."
            )

        values = (self.fps, self.width, self.height)
        if any(v is not None for v in values) and any(v is None for v in values):
            raise ValueError(
                "For `fps`, `width`, and `height`, either all must be set or none must be set."
            )
